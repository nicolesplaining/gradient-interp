import json

import pytest

from gradient_interp.paper_magnitude_data import (
    Components,
    canonical,
    choose,
    cluster_prompts,
    historical_inputs,
    prompt_strings,
    render_source,
    token_counts,
    validate,
)


def test_historical_extractor_reads_inputs_not_answers(tmp_path):
    root = tmp_path / "runs" / "old"
    root.mkdir(parents=True)
    value = {
        "prompt": "User: Summarize this.\nAssistant:",
        "completion": "Never a prompt.",
        "nested": [
            {"role": "user", "content": "Second question?"},
            {"role": "assistant", "content": "Not a prompt."},
        ],
    }
    (root / "data.json").write_text(json.dumps(value))
    (root / "results.json").write_text(json.dumps({"prompt": "Do not read this output."}))
    result = historical_inputs(tmp_path, "fixture")
    assert "summarize this" in result["prompts"]
    assert "second question" in result["prompts"]
    assert len(result["files"]) == 1
    assert all("not a prompt" not in p for p in result["prompts"])
    assert "never a prompt" not in result["prompts"]
    assert canonical("ＣＡＴ  naïve!") == "cat naïve"
    assert list(prompt_strings({"instruction": "Count", "input": "cats", "output": "3"})) == [
        "Count",
        "Count\n\ncats",
    ]


def test_duplicate_components_include_history_and_instruction_groups():
    rows = [
        {"prompt": "Describe orbit speed earth", "instruction": "Describe orbit speed earth"},
        {
            "prompt": "Describe orbit speed earth moon",
            "instruction": "Describe orbit speed earth moon",
        },
        {"prompt": "Translate\n\ncat", "instruction": "Translate"},
        {"prompt": "Translate\n\ndog", "instruction": "Translate"},
    ]
    labels, forbidden, stats = cluster_prompts(
        rows, ["Describe orbit speed earth"], threshold=0.5, chunk=2
    )
    assert labels[0] == labels[1] and labels[0] in forbidden
    assert labels[2] == labels[3]
    assert stats["near_duplicate_edges"] > 0
    components = Components(4)
    components.join(0, 1)
    components.join(2, 3)
    components.join(1, 3)
    assert {components.find(i) for i in range(4)} == {0}


def test_selection_is_cluster_disjoint_deterministic_and_length_bounded():
    rows = [
        dict(
            source_id=f"row:{i}",
            source_row=i,
            prompt=f"Question {i}",
            instruction=f"Question {i}",
            answer="Answer",
            full_tokens=20,
            completion_tokens=4,
        )
        for i in range(20)
    ]
    rows[0]["full_tokens"] = 501
    labels = [0, 0] + list(range(2, 20))
    counts = dict(batches=2, documents_per_batch=2, calibration=1, evaluation=2)
    data, stats = choose(rows, labels, {4, 5}, counts)
    other, _ = choose(rows, labels, {4, 5}, counts)
    assert data == other
    assert stats["selected_rows"] == 7
    validate(data, counts)
    all_rows = (
        [r for b in data["batches"] for r in b["documents"]]
        + data["calibration"]
        + data["evaluation"]
    )
    assert not {r["source_row"] for r in all_rows}.intersection({0, 4, 5})
    data["evaluation"][0] = data["calibration"][0]
    with pytest.raises(AssertionError):
        validate(data, counts)
    with pytest.raises(ValueError, match="eligible clusters"):
        choose(rows[:1], labels[:1], set(), counts)


def test_original_template_prefix_mask_length_and_source_ids():
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "prompt" if add_generation_prompt else "full"

        def __call__(self, texts, add_special_tokens):
            assert texts == ["full", "prompt"] and not add_special_tokens
            return {"input_ids": [[2, 8, 9, 4, 1], [2, 8, 9, 1]]}

    assert token_counts(Tokenizer(), {"prompt": "Q", "answer": "A"}) == (5, 2)
    rows = render_source(
        [
            {"instruction": " ", "input": "", "output": "A"},
            {"instruction": " Q ", "input": " C ", "output": " A "},
        ]
    )
    assert len(rows) == 1 and rows[0]["source_row"] == 1
    assert rows[0]["prompt"] == "Q\n\nC" and rows[0]["source_id"].endswith(":row:1")
