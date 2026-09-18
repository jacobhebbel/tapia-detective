"""Create trust dictionary based on co-location, excluding trust in killer.

For each character, produce a list of dictionaries, each dict corresponds to
another character: {other_character: 1 if trusted else 0}.
A character trusts another if they are in the same room and the other is not the killer.
"""

from House_Generator import generate_house
from CharClass import build_characters

def compute_trust_friends() -> dict[str, list[dict]]:
    """Return trust dictionary as described."""
    # Generate house and characters
    house = generate_house()
    chars = build_characters(house)  # name -> [role, location, trusts, personality]

    # Determine killer: look for role starting with "Killer"
    killer_name = None
    for name, details in chars.items():
        role = details[0]
        if role.startswith("Killer"):
            killer_name = name
            break
        if role.startswith("Victim"):
                    victim_name = name
                    break

    # Prepare list of character names for consistent order
    char_names = list(chars.keys())

    # Build trust dict
    trust_dict: dict[str, list[dict]] = {}
    for name in char_names:
        location = chars[name][1]  # current location
        trust_list = []
        for other in char_names:
            if other == name:
                # Skip self? We'll still include maybe as 0? We'll skip self to keep list of others.
                continue
            other_location = chars[other][1]
            # Trust if same room and other is not killer
            trusted = 1 if (location == other_location and (other != killer_name and other != victim_name)) else 0
            trust_list.append({other: trusted})
        trust_dict[name] = trust_list
    return trust_dict

if __name__ == "__main__":
    trust_friends = compute_trust_friends()
    for char, lst in trust_friends.items():
        print(f"{char}:")
        for d in lst:
            # each dict has single item
            for other, val in d.items():
                print(f"  {other}: {val}")