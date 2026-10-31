import json

import pytest

pytest.importorskip("google.genai")

from google.genai import errors as genai_errors

import src.assistant as assistant
from tests.helpers import seed_nutrition


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.models = Obj(generate_content=self.generate_content)

    def generate_content(self, model, contents, config):
        self.calls.append(Obj(model=model, contents=contents, config=config))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeAPIError(genai_errors.APIError):
    def __init__(self, code):
        Exception.__init__(self, str(code))
        self.code = code
        self.message = "test error"


def tool_turn(*calls):
    content = Obj(role="model", parts=["opaque model parts"])
    response = Obj(
        function_calls=[Obj(name=name, args=args) for name, args in calls],
        candidates=[Obj(content=content)],
        text=None,
    )
    return response, content


def answer(text):
    return Obj(function_calls=None, candidates=[Obj(content=Obj(role="model", parts=[]))], text=text)


USER_QUESTION = [{"role": "user", "content": "How am I doing?"}]


def test_function_declarations_build_a_valid_request_config():
    from google.genai import types

    declarations = assistant._function_declarations()
    config = types.GenerateContentConfig(
        system_instruction=assistant.SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=declarations)],
    )
    assert config is not None
    assert all("parameters" not in d or d["parameters"]["properties"] for d in declarations)


def test_tool_loop_runs_tools_then_returns_the_final_answer(temp_db):
    seed_nutrition(days=60)
    turn, model_content = tool_turn(("predict_weight", {"days": 7}), ("detect_plateau", {}))
    client = FakeClient([turn, answer("Your weight looks stable.")])

    result = assistant.chat(USER_QUESTION, client=client)

    assert result == {"reply": "Your weight looks stable.", "tools_used": ["predict_weight", "detect_plateau"]}
    second_call = client.calls[1].contents
    assert second_call[-2] is model_content
    function_parts = second_call[-1].parts
    assert len(function_parts) == 2
    for part in function_parts:
        payload = part.function_response.response["result"]
        json.dumps(payload)
        assert "error" not in payload


def test_assistant_history_is_sent_with_the_model_role(temp_db):
    client = FakeClient([answer("Fine.")])
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and now?"},
    ]
    assistant.chat(history, client=client)
    assert [content.role for content in client.calls[0].contents] == ["user", "model", "user"]


def test_tool_loop_is_capped(temp_db):
    endless = [tool_turn(("get_model_explanation", {}))[0] for _ in range(20)]
    client = FakeClient(endless)
    result = assistant.chat(USER_QUESTION, client=client)
    assert len(client.calls) == assistant.MAX_TOOL_ROUNDS
    assert "could not finish" in result["reply"]


def test_empty_model_text_gets_a_fallback_message(temp_db):
    result = assistant.chat(USER_QUESTION, client=FakeClient([answer(None)]))
    assert result["reply"]


def test_api_errors_are_translated_to_friendly_status_codes(temp_db):
    expected = {429: 429, 404: 503, 403: 503, 500: 502}
    for api_code, status in expected.items():
        with pytest.raises(assistant.AssistantError) as caught:
            assistant.chat(USER_QUESTION, client=FakeClient([FakeAPIError(api_code)]))
        assert caught.value.status_code == status


def test_missing_api_key_is_reported(temp_db, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(assistant.AssistantError) as caught:
        assistant.chat(USER_QUESTION)
    assert caught.value.status_code == 503


def test_weekly_report_makes_one_request_with_all_model_results(temp_db):
    seed_nutrition(days=60)
    temp_db.add_workout_set("2026-03-01", "Squat", 100.0, 5, None)
    client = FakeClient([Obj(text="  A short report.  ")])

    result = assistant.weekly_report(client=client)

    assert result == {"report": "A short report."}
    assert len(client.calls) == 1
    prompt = client.calls[0].contents
    payload = json.loads(prompt[prompt.index("Model results:\n") + len("Model results:\n"):])
    assert set(payload) == {
        "weight_trend",
        "forecast_14_days",
        "maintenance",
        "unusual_weigh_ins_last_7_days",
        "training",
    }
    assert "Squat" in payload["training"]


def test_weekly_report_still_works_with_no_data(temp_db):
    client = FakeClient([Obj(text="Nothing to report yet.")])
    assert assistant.weekly_report(client=client)["report"] == "Nothing to report yet."