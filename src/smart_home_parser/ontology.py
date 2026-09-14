from dataclasses import dataclass

NONE = "none"

INTENT = [
    "device_control",
    "unsupported",
]

ACTIONS = [
    "turn_on",
    "turn_off",
    "set_temperature",
    "open",
    "close",
    "set_volume",
    "play",
    "pause",
    "lock",
]

DEVICES = [
    "light",
    "thermostat",
    "speaker",
    "door_lock",
    "garage_door",
    "blinds",
    "fan",
]

LOCATIONS = [
    "kitchen",
    "living_room",
    "bedroom",
    "office",
    "bathroom",
    "front_door",
    "garage",
    "whole_home",
    "unknown",
]

UNITS = [
    NONE,
    "percent",
    "fahrenheit",
]

VALUE_CLASSES = [
    NONE,
    *[str(value) for value in range(0, 101)],  # percent values from 0 to 100
]

LABEL_FIELDS = (
    "intent",
    "action",
    "device",
    "location",
    "value",
    "unit",
)


@dataclass(frozen=True)
class CommandLabels:
    intent: str
    action: str
    device: str
    location: str
    value: str = NONE
    unit: str = NONE


VALID_ACTIONS_BY_DEVICE = {
    "light": {"turn_on", "turn_off", "set_brightness"},
    "thermostat": {"turn_on", "turn_off", "set_temperature"},
    "speaker": {"turn_on", "turn_off", "set_volume", "play", "pause"},
    "door_lock": {"lock"},
    "garage_door": {"open", "close"},
    "blinds": {"open", "close"},
    "fan": {"turn_on", "turn_off"},
}


def is_valid_command(labels: CommandLabels) -> bool:
    if labels.intent == "unsupported":
        return (
            labels.action == NONE
            and labels.device == NONE
            and labels.location == NONE
            and labels.value == NONE
            and labels.unit == NONE
        )

    if labels.intent != "device_control":
        return False

    valid_actions = VALID_ACTIONS_BY_DEVICE.get(labels.device, set())

    if labels.action not in valid_actions:
        return False

    if labels.action == "set_brightness":
        return labels.value != NONE and labels.unit == "percent"

    if labels.action == "set_temperature":
        return labels.value != NONE and labels.unit == "fahrenheit"

    if labels.action == "set_volume":
        return labels.value != NONE and labels.unit == "percent"

    return labels.value == NONE and labels.unit == NONE
