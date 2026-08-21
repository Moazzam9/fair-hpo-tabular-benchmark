from fair_hpo.evaluation.data import (
    get_fairness_groups,
    get_outer_data_with_sensitive,
)


def test_adult_sensitive_attribute():
    (
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_train,
        sensitive_test,
    ) = get_outer_data_with_sensitive(
        "adult",
        1,
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    assert len(sensitive_train) == len(X_train)
    assert len(sensitive_test) == len(X_test)

    assert sensitive_train.name == "sex"
    assert sensitive_test.name == "sex"

    groups = get_fairness_groups("adult")

    assert groups["sensitive_attribute"] == "sex"
    assert groups["privileged_group"] == "Male"
    assert groups["unprivileged_group"] == "Female"
