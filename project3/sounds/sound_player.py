import requests
import numpy as np
from playsound import playsound


class soundplayer():
    def __init__(self, *args, **kwargs):
        self.sounds = sounds = ['sounds/Not_Gay_Sex.mp3', 'sounds/Objection_Heresay.mp3',
                                'sounds/Rock_Flag_and_Eagle.mp3',
                                'sounds/The_good_lords_goin_down_on_me.mp3',
                                'sounds/my-man.mp3', 'idubbbz-im-gay-free-download.mp3']
        self.sound_dict = {}
        self.url_reg = r"(http(?:s){0,1}://[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]\
                         {1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))"

    def __call__(self):
        if *args.shape != 0:
            try sound = self.sound_dict{*args}
            playsound(sounds[np.random.randint(0,6)])

    def update_list(self)
        urls = input('insert url to download')
        
