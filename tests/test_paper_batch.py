from gradient_interp.paper_batch import STYLES, alternatives, generated_checks, make_data


def test_controlled_content_crosses_format_and_length_without_changing_tags():
    tags = ["ZABC123", "ZDEF456", "Z789ABC"]
    choices = alternatives(tags)
    assert len(choices) == 4
    for answer in choices:
        assert generated_checks(answer, tags)["exact_content_and_order"]
    assert [generated_checks(x, tags)["numbered"] for x in choices] == [False, True, False, True]
    assert len(choices[2]) > len(choices[0]) and len(choices[3]) > len(choices[1])


def test_wrong_duplicate_missing_extra_or_reordered_tags_fail_content_check():
    tags = ["ZABC123", "ZDEF456", "Z789ABC"]
    for answer in [
        tags[0],
        " ".join(tags[::-1]),
        " ".join(tags + [tags[0]]),
        " ".join(tags + ["Z000000"]),
    ]:
        assert not generated_checks(answer, tags)["exact_content_and_order"]


def test_development_train_calibration_evaluation_have_distinct_prompts_and_tags():
    data = make_data()
    train = [doc for batch in data["batches"] for doc in batch["documents"]]
    all_rows = train + data["calibration"] + data["evaluation"]
    assert len(all_rows) == len({r["prompt"] for r in all_rows}) == 96
    assert len({tag for r in all_rows for tag in r["tags"]}) == 288
    assert [b["style"] for b in data["batches"]].count(STYLES[0]) == 4
    assert make_data() == data
