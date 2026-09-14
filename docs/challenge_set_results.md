# Challenge Set Results

## Summary

- Model version: 0.1.0
- Examples: 50
- Exact-match accuracy: 0.58 (29/50)
- Unsupported recall: 0.65 (11/17)
- Intent accuracy: 0.74 (37/50)
- Status: exact match is within the target range; unsupported recall remains below the 0.70-0.95 target

The evaluation ran locally on CPU using the existing Colab-trained artifact.
PyTorch emitted a non-fatal nested-tensor warning during model construction.

## Failure Cases

| Command | Expected behavior | Model prediction | Root cause | Planned fix |
|---|---|---|---|---|
| "Could you make it brighter where I'm cooking?" | Set kitchen light brightness | Set bedroom thermostat temperature | Implicit brightness intent and cooking-to-kitchen phrase were unseen | Add implicit brightness and contextual location paraphrases |
| "Kill the lamps upstairs." | Turn off bedroom lights | Unsupported, with light device only | "Kill," "lamps," and "upstairs" aliases were unseen | Add action, device, and location alias families |
| "Put the lounge lights at half power." | Set living-room brightness to 50 percent | Unsupported | "Lounge," "half power," and the brightness paraphrase were unseen | Add living-room aliases and spoken/relative brightness phrases |
| "Quiet the office music down to 20%." | Set speaker volume to 20 percent | Play speaker music at value 20 with no unit | Volume reduction wording was confused with play intent | Add volume-down paraphrases and balance set-volume examples |
| "Can you play something in the kitchen?" | Play kitchen speaker | Play with no device | "Something" did not provide a learned speaker cue | Add speaker aliases and implicit speaker-play examples |
| "Pause whatever is playing in my room." | Pause bedroom speaker | Unsupported | "Whatever is playing" and "my room" were unseen pause/location phrasing | Add pause paraphrases and bedroom aliases |
| "Let sunlight into the bathroom." | Open bathroom blinds | Set bathroom thermostat temperature | Metaphorical blinds-opening phrase was unseen | Add blinds aliases such as sunlight, shades, and raise/open phrasing |
| "Make the fan go." | Turn on fan with unknown location | Turn on bedroom fan with fahrenheit unit | Locationless fan command and unknown-location class were absent from the artifact | Add locationless device examples and train the unknown location/unit behavior |
| "Stop the fan in the garage." | Turn off garage fan | Turn on garage fan | Stop/off distinction was not learned reliably for fans | Add stop/disable fan paraphrases and targeted contrast examples |
| "Secure the entrance." | Lock the front door | Unsupported, with otherwise correct lock fields | Short entrance synonym was classified as unsupported | Add entrance/security lock paraphrases |
| "Play the kitchen lights." | Unsupported | Turn off kitchen light | Invalid device/action combination resembled a valid light command | Add invalid combinations with unsupported intent and strengthen intent gating |
| "Turn the front door up to 80 percent." | Unsupported | Set bathroom light brightness to 80 percent | Invalid door/value combination resembled brightness syntax | Add unsupported commands containing misleading numeric device-control syntax |
| "Set the refrigerator to seventy degrees." | Unsupported | Set bedroom thermostat to 70 degrees | Refrigerator was not represented as an out-of-domain device | Add appliance temperature negatives and unsupported device aliases |
| "Start the office fan." | Turn on office fan | Turn off office fan | Start/stop fan wording was confused | Add paired fan start/stop examples |
| "Play a song in the bedroom." | Play bedroom speaker | Unsupported | Novel play wording was rejected as out of domain | Add song/content paraphrases for speaker playback |
| "Start a song on the bathroom speaker." | Play bathroom speaker | Unsupported | Speaker playback wording was unseen | Add explicit speaker and content paraphrases |
| "Please pause whatever is playing in the kitchen." | Pause kitchen speaker | Unsupported | Longer pause paraphrase was classified as unsupported | Add pause context and filler variations |
| "Turn on the fan." | Turn on fan with unknown location | Turn on garage fan | Locationless command was forced to a learned room | Add more locationless commands with `location="unknown"` |
| "Set the oven to 400 degrees." | Unsupported | Set bedroom thermostat to 69 degrees | Appliance temperature request matched the thermostat pattern | Add oven and appliance hard negatives |
| "Play the garage lights." | Unsupported | Turn on garage light | Invalid play/light combination resembled a valid light command | Add invalid device/action combinations across all devices |
| "Set the front door to 50 percent." | Unsupported | Set bathroom light brightness to 50 percent | Door plus percentage syntax triggered brightness classification | Add misleading numeric unsupported examples and device constraints |

## Next Fixes

1. Add locationless commands labeled with `location="unknown"`, including fan and light examples.
2. Add the failed aliases and paraphrase families to the generator.
3. Add hard negative examples that combine valid verbs with invalid devices and appliances.
4. Regenerate the dataset and retrain a new artifact version.
5. Re-run this 50-example challenge set unchanged and compare exact match, unsupported recall, and the confusion matrix.