"""Independent pinned-source encoding and Hugging Face BPE conformance fixtures."""

import hashlib
import importlib.util
import json
import os
import random
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "locks/metadata/deepseek-ai/DeepSeek-V4.1-Flash/2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
)
EXE = Path(
    os.environ.get(
        "RIA_FRONTEND_FIXTURE", str(ROOT / "build/ria/tests/test_frontend_contracts")
    )
).resolve()


def native(args, value, success=True):
    result = subprocess.run(
        [str(EXE), *args],
        input=value.encode(),
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert (result.returncode == 0) == success, result.stderr.decode()
    return result.stdout.decode()


@pytest.fixture(scope="module")
def encoding():
    spec = importlib.util.spec_from_file_location(
        "pinned_encoding", SOURCE / "encoding/encoding.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_bpe():
    from tokenizers import Tokenizer

    path = SOURCE / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    rng = random.Random(718)
    alphabet = list("aAZ09!?, .\r\n\t中文日本語あいうえおéêαβ🙂∑")
    texts = [
        "",
        "hello world",
        "01234567890123456789",
        "  \n\nhello\tworld \r\n",
        "中文日本語あいうえお",
        "é\u0301🙂\x00\u200d",
        "<｜User｜>hi<｜Assistant｜><think>为什么</think>",
        "<<" * 10000,
    ]
    texts += ["".join(rng.choice(alphabet) for _ in range(1000)) for _ in range(4)]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for text in texts:
        actual = json.loads(native(["tokenize", str(path), sha], text))
        assert actual == tokenizer.encode(text, add_special_tokens=False).ids
    native(["tokenize", str(path), "0" * 64], "hello", success=False)


def test_native_prompt(encoding):
    tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "n": {"type": "number", "default": 1e-7},
                },
            },
        },
    }
    cases = [
        [{"role": "user", "content": "Hello"}],
        [
            {"role": "system", "content": "Be precise"},
            {"role": "user", "content": "Hi"},
        ],
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [
            {"role": "user", "content": "a", "task": None},
            {"role": "assistant", "content": "b", "reasoning_content": "because"},
            {"role": "user", "content": "c"},
        ],
        [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c", "reasoning_content": "d"},
            {"role": "system", "content": "e"},
        ],
        [
            {"role": "system", "content": "tools", "tools": [tool]},
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "check",
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"city": "北京", "n": 1.0, "a": [True, None]},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "x", "content": "sunny"},
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                    {"type": "text", "text": "b"},
                ],
            }
        ],
    ]
    for argument in [
        "not JSON",
        "123",
        json.dumps(json.dumps({"x": "é"})),
        [1, "x"],
        None,
    ]:
        cases.append(
            [
                {"role": "user", "content": "a"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "x", "arguments": argument},
                        }
                    ],
                },
                {"role": "user", "content": "b"},
            ]
        )
    cases.append(
        [{"role": "system", "content": "a\x00b"}, {"role": "user", "content": []}]
    )
    cases.append(
        [
            {
                "role": "assistant",
                "content": "a",
                "tool_calls": [
                    {
                        "id": "two",
                        "type": "function",
                        "function": {"name": "x", "arguments": {}},
                    },
                    {
                        "id": "one",
                        "type": "function",
                        "function": {"name": "x", "arguments": {}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "one", "content": "first"},
            {"role": "tool", "tool_call_id": "two", "content": "second"},
            {"role": "user", "content": "continue"},
        ]
    )
    for task in ["action", "query", "authority", "domain", "title", "read_url"]:
        cases.append([{"role": "user", "content": "a", "task": task}])
    for thinking in [False, True]:
        for drop in [False, True]:
            for messages in cases:
                body = {
                    "messages": messages,
                    "thinking": thinking,
                    "drop": drop,
                    "effort": 50,
                }
                actual = native(["render"], json.dumps(body, ensure_ascii=False))
                expected = encoding.encode_messages(
                    messages,
                    thinking_mode="thinking" if thinking else "chat",
                    drop_thinking=drop,
                    reasoning_effort=50,
                )
                assert actual == expected
    native(
        ["render"],
        '{"messages":[{"role":"user","content":"<｜deepseek_image｜>"}]}',
        success=False,
    )


def test_encoded_argument_boundaries():
    for value in [
        '{"x":1,"x":2}',
        "NaN",
        "Infinity",
        "-Infinity",
        '{"x":NaN}',
        '{"x":1e999}',
    ]:
        messages = [
            {"role": "user", "content": "a"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "f", "arguments": value}}
                ],
            },
        ]
        native(["render"], json.dumps({"messages": messages}), success=False)


def test_native_completion(encoding):
    eos = "<｜end▁of▁sentence｜>"
    calls = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="ns::weather">\n<｜DSML｜ parameter name="city" string="true">北京</｜DSML｜ parameter>\n<｜DSML｜ parameter name="n" string="false">1.0</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    for body in ["hello", "a\x00b", "", "before" + calls, calls]:
        for thinking in [False, True]:
            text = ("because</think>" if thinking else "") + body + eos
            mode = "thinking" if thinking else "chat"
            assert json.loads(
                native(["completion", mode], text)
            ) == encoding.parse_message_from_completion_text(text, mode)
    for text in [
        "hello",
        "hello" + eos + "trailing",
        "<think>bad" + eos,
        calls.replace('name="n"', 'name="city"') + eos,
        calls[:-1] + eos,
    ]:
        native(["completion", "chat"], text, success=False)


def test_native_http_boundary():
    good = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer fixture-token\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n"
    assert native(["header"], good) == "POST /v1/chat/completions 2\n"
    assert (
        native(
            ["header"],
            "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer fixture-token\r\n\r\n",
        )
        == "GET /v1/models 0\n"
    )
    bad = [
        good.replace("fixture-token", "wrong-token"),
        good.replace("Authorization: Bearer fixture-token\r\n", ""),
        good.replace("Host: localhost", "Host: localhost\r\nHost: other"),
        good.replace(
            "Authorization: Bearer fixture-token",
            "Authorization: Bearer fixture-token\r\nAuthorization: Bearer fixture-token",
        ),
        good.replace("Content-Length: 2", "Content-Length: 2\r\nContent-Length: 2"),
        good.replace("Content-Length: 2", "Content-Length: 02"),
        good.replace("Content-Length: 2", "Content-Length: -1"),
        good.replace("Content-Length: 2", "Content-Length: 18446744073709551616"),
        good.replace("Content-Length: 2", "Content-Length: 4097"),
        good.replace("Content-Length: 2", "Content-Length: 0"),
        good.replace(
            "Content-Length: 2", "Transfer-Encoding: chunked\r\nContent-Length: 2"
        ),
        good.replace("Content-Type: application/json", "Content-Type: text/plain"),
        good.replace("Host:", " Host:"),
        good.replace("HTTP/1.1", "HTTP/1.0"),
        good.replace("POST", "GET"),
        good.replace("POST", "DELETE"),
        good.replace("/v1/chat/completions", "/v1/chat#fragment"),
        good.replace("\r\n", "\n"),
        good[:-1],
        good.replace("Host: localhost", "Host: local\x00host"),
        good.replace("Host: localhost", "Host: local\x7fhost"),
    ]
    for value in bad:
        native(["header"], value, success=False)


def test_source_byte_decoder():
    cases = [
        b"",
        b"a\x00b",
        "中文🙂".encode(),
        b"\xc0\xaf",
        b"\xe0\x80\x80",
        b"\xed\xa0\x80",
        b"\xf4\x90\x80\x80",
        b"a\xe2\x82Z",
        b"\xf0\x90\x80",
    ]
    rng = random.Random(7118)
    cases += [bytes(rng.randrange(256) for _ in range(8192)) for _ in range(8)]
    for value in cases:
        result = subprocess.run(
            [str(EXE), "utf8"],
            input=value,
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert result.stdout.decode() == value.decode(errors="replace")


def test_authenticated_measurement_option():
    assert native(["measurements"], "{}").strip() == "false"
    assert native(["measurements"], '{"stream":true}').strip() == "false"
    assert (
        native(["measurements"], '{"stream":true,"ria_measurements":true}').strip()
        == "true"
    )
    assert (
        native(["measurements"], '{"stream":false,"ria_measurements":false}').strip()
        == "false"
    )
    for value in (None, 0, 1, "true", [], {}):
        native(
            ["measurements"],
            json.dumps({"stream": True, "ria_measurements": value}),
            success=False,
        )
    native(["measurements"], '{"stream":false,"ria_measurements":true}', success=False)


def test_native_process_sigpipe_policy():
    assert native(["process-policy"], "").strip() == "protected"


@pytest.mark.parametrize(
    "token,expected",
    [
        (b"fixture-token", 13),
        (b"fixture-token\n", 13),
        (b"fixture-token\r\n", 13),
        (b"x" * 4096, 4096),
        (b"x" * 4096 + b"\n", 4096),
        (b"x" * 4096 + b"\r\n", 4096),
    ],
)
def test_native_bearer_regular_file(tmp_path, token, expected):
    path = tmp_path / "api.token"
    path.write_bytes(token)
    assert native(["bearer", str(path)], "").strip() == str(expected)


@pytest.mark.parametrize(
    "token",
    [
        b"",
        b"\n",
        b"\r\n",
        b"fixture token",
        b"fixture\ttoken",
        b"token\r",
        b"token\n\n",
        b"token\x00suffix",
        b"token\x7f",
        b"token\xff",
        b"x" * 4097,
        b"x" * 4097 + b"\r\n",
    ],
)
def test_native_bearer_invalid_file_clears_output(tmp_path, token):
    path = tmp_path / "api.token"
    path.write_bytes(token)
    native(["bearer", str(path)], "", success=False)


def test_native_bearer_rejects_nonregular_without_blocking(tmp_path):
    path = tmp_path / "token.fifo"
    os.mkfifo(path, 0o600)
    # An unconnected FIFO previously blocked startup inside fopen. Reject it
    # on the opened descriptor; do not wait for a writer or consume its bytes.
    result = subprocess.run(
        [str(EXE), "bearer", str(path)],
        input=b"",
        capture_output=True,
        timeout=1,
        check=False,
    )
    assert result.returncode == 1, result.stderr.decode()
    assert not result.stdout
    for rejected in (tmp_path, tmp_path / "missing.token", Path("/dev/null")):
        native(["bearer", str(rejected)], "", success=False)
    regular = tmp_path / "api.token"
    regular.write_bytes(b"fixture-token")
    link = tmp_path / "token.link"
    link.symlink_to(regular)
    native(["bearer", str(link)], "", success=False)


def test_native_nll_large_common_offsets():
    import math
    import struct

    losses = json.loads(native(["nll"], ""))
    assert losses[:5] == pytest.approx([math.log(129280)] * 5, rel=1e-15)
    float_max = struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0]
    assert losses[5] == 2 * float_max
