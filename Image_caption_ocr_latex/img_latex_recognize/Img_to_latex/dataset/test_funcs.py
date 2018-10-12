import hashlib
list1 = [
    '\angle D = 9 0 ^ { \circ } .', '\angle E = 9 0 ^ { \circ } .',
    '\angle F = 9 0 ^ { \circ } .', '\angle C = 6 0 ^ { \circ } .',
    '\angle C = 7 0 ^ { \circ } .', '\angle C = 8 0 ^ { \circ } .',
    '\angle C = 9 1 ^ { \circ } .', '\angle C = 9 2 ^ { \circ } .',
    '\angle C = 9 3 ^ { \circ } .'
]
for i in list1:
    name = hashlib.sha256(i.encode('utf-8')).hexdigest()
    print(name)
