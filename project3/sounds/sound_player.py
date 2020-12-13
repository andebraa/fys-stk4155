import numpy as np
from playsound import playsound

class soundplayer():
    def __init__(self, *args, **kwargs):
        sounds = ['sounds/Not_Gay_Sex.mp3', 'sounds/Objection_Heresay.mp3',
                  'sounds/Rock_Flag_and_Eagle.mp3', 'sounds/The_good_lords_goin_down_on_me.mp3',
                  'sounds/my-man.mp3', 'idubbbz-im-gay-free-download.mp3']

    def random(self):
            sounds = ['sounds/Not_Gay_Sex.mp3', 'sounds/Objection_Heresay.mp3',
                      'sounds/Rock_Flag_and_Eagle.mp3', 'sounds/The_good_lords_goin_down_on_me.mp3',
                      'sounds/my-man.mp3', 'idubbbz-im-gay-free-download.mp3']

            playsound(sounds[np.random.randint(0,6)])
