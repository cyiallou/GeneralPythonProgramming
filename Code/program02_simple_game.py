"""Simple text based game.

You could create different classes (e.g., elves, orcs, humans), fight them,
make them fight each other, and so on.

Source: Object-Oriented Programming lesson from https://www.sololearn.com
        (Python 3 Tutorial)
"""

# =====================================
#
# CLASS DEFINITIONS
#
# =====================================


class GameObject:
    class_name = ''
    desc = ''
    objects = {}

    def __init__(self, name):
        self.name = name
        GameObject.objects[self.class_name] = self

    def get_desc(self):
        return f'Name: {self.name}\n Type: {self.class_name}\n Description: {self.desc}'


class Goblin(GameObject):
    def __init__(self, name):
        self.class_name = 'goblin'
        self.health = 3
        self._desc = 'An unearthly creature.'
        super().__init__(name)

    @property
    def desc(self):
        if self.health >= 3:
            return self._desc
        elif self.health == 2:
            health_line = 'It has a wounded right arm.'
        elif self.health == 1:
            health_line = 'Its left leg has been chopped off!'
        elif self.health <= 0:
            health_line = 'It is dead.'
        return f'{self._desc}\n {health_line}'

    @desc.setter
    def desc(self, value):
        self._desc = value


# Create characters
goblin = Goblin('Gremlin')


# =====================================
#
# HELPER FUNCTIONS
#
# =====================================
def hit(noun):
    if noun in GameObject.objects:
        thing = GameObject.objects[noun]
        if type(thing) == Goblin:
            thing.health = thing.health - 1
            if thing.health <= 0:
                msg = 'You killed the goblin!'
            else:
                msg = f'You hit the {thing.class_name}'
    else:
        msg = f'There is no {noun} here.'
    return msg


def get_input():
    command = input(': ').split()
    verb_word = command[0]
    if verb_word in verb_dict:
        verb = verb_dict[verb_word]
    else:
        print(f'Unknown verb {verb_word}')
        return
    if len(command) >= 2:
        noun_word = command[1]
        print(verb(noun_word))
    else:
        print(verb('nothing'))


def say(noun):
    return f'You said "{noun}"'


def examine(noun):
    if noun.lower() in GameObject.objects:
        return GameObject.objects[noun.lower()].get_desc()
    else:
        return f'There is no "{noun}" here.'


# add the verbs with the associated functions here
verb_dict = {
        'say': say,
        'examine': examine,
        'hit': hit,
        }


while True:
    get_input()
