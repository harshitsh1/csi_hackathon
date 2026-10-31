"""Label vocabularies. Labels come from the source file/folder names and are used ONLY as
supervision targets (never as model input features)."""
import re

UNIFIED_LABELS = ["EMPTY", "LYING", "SIT", "STAND", "WALK", "SITDOWN", "STANDUP", "FALL"]
SHARED = ["SIT", "STAND", "WALK"]
ROOM_LABELS = {"A": ["SIT", "STAND", "WALK", "SITDOWN", "STANDUP", "FALL"],
               "B": ["EMPTY", "LYING", "SIT", "STAND", "WALK"]}
EXCLUSIVE = {"A": ["SITDOWN", "STANDUP", "FALL"], "B": ["EMPTY", "LYING"]}

A_MAP = {"fall": "FALL", "sit": "SIT", "sitdown": "SITDOWN", "stand": "STAND",
         "standup": "STANDUP", "walk": "WALK"}
B_MAP = {"empty": "EMPTY", "lying": "LYING", "sitting": "SIT", "standing": "STAND",
         "walking": "WALK"}


def other_room(r):
    return "B" if r == "A" else "A"


def label_room_a(path):
    """Folder name if it is a class folder, else the alphabetic prefix of the file name."""
    p = path.parent.name.lower()
    if p in A_MAP:
        return A_MAP[p]
    m = re.match(r"[a-z]+", path.stem.lower())
    return A_MAP.get(m.group(0)) if m else None


def label_room_b(path):
    return B_MAP.get(path.stem.split("_")[0].lower())
