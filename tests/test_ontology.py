from smart_home_parser.ontology import NONE, CommandLabels, is_valid_command


def test_light_brightness_command_is_valid() -> None:
    labels = CommandLabels(
        intent="device_control",
        action="set_brightness",
        device="light",
        location="kitchen",
        value="30",
        unit="percent",
    )

    assert is_valid_command(labels) is True


def test_invalid_device_action_pair_is_rejected() -> None:
    labels = CommandLabels(
        intent="device_control",
        action="play",
        device="thermostat",
        location="bedroom",
    )

    assert is_valid_command(labels) is False


# initial values for thermostat temperature setting command
def test_temperature_requires_fahrenheit_value() -> None:
    labels = CommandLabels(
        intent="device_control",
        action="set_temperature",
        device="thermostat",
        location="bedroom",
        value="72",
        unit="fahrenheit",
    )

    assert is_valid_command(labels) is True


def test_unsupported_command_requires_empty_device_fields() -> None:
    labels = CommandLabels(
        intent="unsupported",
        action=NONE,
        device=NONE,
        location=NONE,
    )

    assert is_valid_command(labels) is True
    assert (
        is_valid_command(
            CommandLabels(
                intent="unsupported",
                action="turn_on",
                device=NONE,
                location=NONE,
            )
        )
        is False
    )
