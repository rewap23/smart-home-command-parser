# dataset generation script for smart home commands

from __future__ import annotations

import json
import random
from pathlib import Path

from smart_home_parser.ontology import CommandLabels, NONE, is_valid_command

# seed the random number generator for reproducibility
SEED = 42
random.seed(SEED)

# locations in the smart home environment
LOCATIONS = [
    "kitchen",
    "living room",
    "bedroom",
    "office",
    "bathroom",
    "garage",
]

LIGHT_ON_TEMPLATES = [
    "turn on the {location} lights",
    "switch on the lights in the {location}",
    "please illuminate the {location}",
    "can you turn the {location} lights on",
]

LIGHT_OFF_TEMPLATES = [
    "turn off the {location} lights",
    "switch off the lights in the {location}",
    "please darken the {location}",
    "can you turn the {location} lights off",
]

BRIGHTNESS_TEMPLATES = [
    "set the {location} lights to {value} percent",
    "dim the lights in the {location} to {value}%",
    "make the {location} lighting {value} percent",
]

TEMPERATURE_TEMPLATES = [
    "set the {location} thermostat to {value} degrees",
    "make the {location} temperature {value} fahrenheit",
    "adjust the thermostat in the {location} to {value}",
]

FAN_ON_TEMPLATES = [
    "turn on the {location} fan",
    "switch on the fan in the {location}",
]

FAN_OFF_TEMPLATES = [
    "turn off the {location} fan",
    "switch off the fan in the {location}",
]

SPEAKER_PLAY_TEMPLATES = [
    "play music on the {location} speaker",
    "start music in the {location}",
    "play some music in the {location}",
]

SPEAKER_PAUSE_TEMPLATES = [
    "pause the music in the {location}",
    "stop music on the {location} speaker",
]

VOLUME_TEMPLATES = [
    "set the {location} speaker volume to {value} percent",
    "make the music volume in the {location} {value}%",
    "set the {location} music volume to {value} percent",
]

LOCK_TEMPLATES = [
    "lock the front door",
    "secure the front door lock",
    "lock the door at the entrance",
]

BLINDS_OPEN_TEMPLATES = [
    "open the {location} blinds",
    "raise the blinds in the {location}",
]

BLINDS_CLOSE_TEMPLATES = [
    "close the {location} blinds",
    "lower the blinds in the {location}",
]

NEGATIVE_TEMPLATES = [
    "what is the weather",
    "send an email to Alex",
    "what time is it",
    "set a reminder for tomorrow",
    "call my sister",
    "send a text to Jordan",
    "read my latest message",
    "what is on my calendar",
    "schedule a meeting for Friday",
    "open my inbox",
    "search the web for coffee shops",
    "look up the nearest restaurant",
    "give me directions to the airport",
    "book a flight to Chicago",
    "reserve a table for two",
    "order a pizza",
    "add milk to my shopping list",
    "buy a birthday present",
    "check the price of a laptop",
    "play the news",
    "tell me a joke",
    "play a podcast",
    "start an audiobook",
    "translate this sentence",
    "define the word curious",
    "calculate fifteen times six",
    "convert dollars to euros",
    "how do you spell necessary",
    "who won the game",
    "when is the next holiday",
    "tell me a fun fact",
    "set an alarm for seven",
    "wake me up at six",
    "track my package",
    "check my bank balance",
    "pay my electricity bill",
    "send a document to Sam",
    "create a note about the project",
    "summarize this article",
    "write a birthday message",
    "draft a reply to Morgan",
    "find a recipe for pasta",
    "what should I cook tonight",
    "recommend a movie",
    "show me today's headlines",
    "start a timer for ten minutes",
    "stop the stopwatch",
    "take a photo",
    "record a voice memo",
    "open the camera",
    "turn on the computer",
]

NUMBER_WORDS = {
    60: "sixty",
    61: "sixty one",
    62: "sixty two",
    63: "sixty three",
    64: "sixty four",
    65: "sixty five",
    66: "sixty six",
    67: "sixty seven",
    68: "sixty eight",
    69: "sixty nine",
    70: "seventy",
    71: "seventy one",
    72: "seventy two",
    73: "seventy three",
    74: "seventy four",
    75: "seventy five",
    76: "seventy six",
    77: "seventy seven",
    78: "seventy eight",
    79: "seventy nine",
    80: "eighty",
}


def make_record(
    command: str,
    labels: CommandLabels,
    template_id: str,
    record_id: str,
) -> dict[str, object]:
    if not is_valid_command(labels):
        raise ValueError(f"Invalid labels: {labels}")

    return {
        "id": record_id,
        "command": command,
        "labels": {
            "intent": labels.intent,
            "action": labels.action,
            "device": labels.device,
            "location": labels.location.replace(" ", "_"),
            "value": labels.value,
            "unit": labels.unit,
        },
        "template_id": template_id,
    }


def command_variants(command: str, template_id: str) -> list[str]:
    variants = {command}
    polite_prefixes = ("please ", "could you ", "would you ")
    filler_prefixes = ("hey assistant, ", "okay assistant, ")
    trailing_fillers = (" right now", " for me")

    for prefix in polite_prefixes:
        variants.add(f"{prefix}{command}")
    for prefix in filler_prefixes:
        variants.add(f"{prefix}{command}")
    for suffix in trailing_fillers:
        variants.add(f"{command}{suffix}")

    for prefix in polite_prefixes + filler_prefixes:
        for suffix in trailing_fillers:
            variants.add(f"{prefix}{command}{suffix}")

    for filler in ("hey assistant, ", "okay assistant, "):
        for polite in ("please ", "could you ", "would you "):
            variants.add(f"{filler}{polite}{command}")

    for location in LOCATIONS:
        replacements = (
            (f"in the {location}", f"in my {location}"),
            (f"the {location}", f"my {location}"),
            (f"the {location}", location),
        )
        for source, replacement in replacements:
            if source in command:
                variants.add(command.replace(source, replacement))

    if template_id.startswith("light_"):
        variants.add(command.replace("lights", "lamps"))
        variants.add(command.replace("lighting", "lamps"))
    if template_id == "thermostat_temperature":
        variants.add(command.replace("thermostat", "AC"))
        variants.add(command.replace("thermostat", "heater"))

    if template_id == "thermostat_temperature":
        for value, word in NUMBER_WORDS.items():
            variants.add(command.replace(str(value), word))

    variants.update((command + ".", command + "!", command + "?"))
    variants.add(command.upper())
    variants.add(command.capitalize())
    return sorted(variants)


def generate_examples() -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    counter = 0

    def add(
        templates: list[str],
        labels: CommandLabels,
        template_id: str,
        substitutions: dict[str, list[str]] | None = None,
    ) -> None:
        nonlocal counter

        substitutions = substitutions or {}
        for template in templates:
            for location in substitutions.get("location", [""]):
                for value in substitutions.get("value", [""]):
                    command = template.format(location=location, value=value)
                    counter += 1
                    examples.append(
                        make_record(
                            command=command,
                            labels=CommandLabels(
                                intent=labels.intent,
                                action=labels.action,
                                device=labels.device,
                                location=location.replace(" ", "_")
                                if location
                                else labels.location,
                                value=value if value else labels.value,
                                unit=labels.unit,
                            ),
                            template_id=template_id,
                            record_id=f"example-{counter:06d}",
                        )
                    )

    add(
        LIGHT_ON_TEMPLATES,
        CommandLabels("device_control", "turn_on", "light", "unknown"),
        "light_turn_on",
        {"location": LOCATIONS},
    )
    add(
        LIGHT_OFF_TEMPLATES,
        CommandLabels("device_control", "turn_off", "light", "unknown"),
        "light_turn_off",
        {"location": LOCATIONS},
    )
    add(
        BRIGHTNESS_TEMPLATES,
        CommandLabels(
            "device_control",
            "set_brightness",
            "light",
            "unknown",
            unit="percent",
        ),
        "light_brightness",
        {
            "location": LOCATIONS,
            "value": [str(value) for value in range(10, 101, 10)],
        },
    )
    add(
        TEMPERATURE_TEMPLATES,
        CommandLabels(
            "device_control",
            "set_temperature",
            "thermostat",
            "unknown",
            unit="fahrenheit",
        ),
        "thermostat_temperature",
        {
            "location": LOCATIONS,
            "value": [str(value) for value in range(60, 81)],
        },
    )
    add(
        FAN_ON_TEMPLATES,
        CommandLabels("device_control", "turn_on", "fan", "unknown"),
        "fan_turn_on",
        {"location": LOCATIONS},
    )
    add(
        FAN_OFF_TEMPLATES,
        CommandLabels("device_control", "turn_off", "fan", "unknown"),
        "fan_turn_off",
        {"location": LOCATIONS},
    )
    add(
        SPEAKER_PLAY_TEMPLATES,
            CommandLabels("device_control", "play", "speaker", "unknown"),
            "speaker_play",
            {"location": LOCATIONS},
    )
    add(
        SPEAKER_PAUSE_TEMPLATES,
            CommandLabels("device_control", "pause", "speaker", "unknown"),
            "speaker_pause",
            {"location": LOCATIONS},
    )
    add(
        VOLUME_TEMPLATES,
        CommandLabels(
                "device_control",
                "set_volume",
                "speaker",
                "unknown",
                unit="percent",
        ),
            "speaker_volume",
        {
            "location": LOCATIONS,
            "value": [str(value) for value in range(10, 101, 10)],
        },
    )
    add(
        LOCK_TEMPLATES,
        CommandLabels("device_control", "lock", "door_lock", "front_door"),
        "door_lock",
    )
    add(
        BLINDS_OPEN_TEMPLATES,
        CommandLabels("device_control", "open", "blinds", "unknown"),
        "blinds_open",
        {"location": LOCATIONS},
    )
    add(
        BLINDS_CLOSE_TEMPLATES,
        CommandLabels("device_control", "close", "blinds", "unknown"),
        "blinds_close",
        {"location": LOCATIONS},
    )

    expanded_examples: list[dict[str, object]] = []
    for example in examples:
        command = str(example["command"])
        template_id = str(example["template_id"])
        for variant in command_variants(command, template_id):
            expanded_examples.append(
                {
                    **example,
                    "id": f"example-{len(expanded_examples) + 1:06d}",
                    "command": variant,
                }
            )

    negative_labels = CommandLabels("unsupported", NONE, NONE, NONE)
    for template_index, template in enumerate(NEGATIVE_TEMPLATES):
        if template == "what is the weather":
            negative_template_id = "negative_weather"
        elif template == "send an email to Alex":
            negative_template_id = "negative_email"
        else:
            negative_template_id = f"negative_{template_index:02d}"
        for variant in command_variants(template, negative_template_id):
            expanded_examples.append(
                make_record(
                    command=variant,
                    labels=negative_labels,
                    template_id=negative_template_id,
                    record_id=f"example-{len(expanded_examples) + 1:06d}",
                )
            )

    random.shuffle(expanded_examples)
    return expanded_examples


def main() -> None:
    output_path = Path("data/raw/smart_home_commands.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = generate_examples()

    with output_path.open("w", encoding="utf-8") as file:
        for record in examples:
            file.write(json.dumps(record) + "\n")

    print(f"Wrote {len(examples)} examples to {output_path}")

if __name__ == "__main__":
    main()