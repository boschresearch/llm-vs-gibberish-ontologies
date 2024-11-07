# Experiment resources related to the paper "Do LLMs Really Adapt to Domains? An Ontology Learning Perspective"
# (ISWC 2024)
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from .translator import Translator
from ..utils.utils import safety_check
from typing import Optional, Callable
import gibberify


class PureGibberishTranslator(Translator):
    def __init__(self, reconfig=False,
                 seed: int = 0):
        if reconfig:
            conf = gibberify.Config()
            conf["gib_langs"]["dark_elv"] = {
                "pool": ["en"],
                "enrich": ["r", "t", "s", "m"],
                "impoverish": ["z", "w", "x"],
                "remove": []
            }
            gibberify.build(conf)
        self.seed = seed
        self.tr = gibberify.Translator('en', 'dark_elv', seed=self.seed)


    def translate(self, word: str, definition: str, pos: str,
                  do_sample: bool = False,
                  max_tries: int = -1,
                  blacklist_fn: Optional[Callable[[str], bool]] = None
                  ) -> str:
        """Make a new nonexistent word based on the definition of this word using GPT2.

        Args:
            word (str): Word to translate.
            definition (str): Definition of the word to translate. This is not used in this translator.
            pos (str): Part of speech of the word to translate (e.g. "noun", "verb"). This is not used in this translator.
            do_sample (bool, optional): Whether to use sampling to generate the new word. This is not used in this translator.
                            Note that if max_tries is greater than 1, this will be set to True after the first try fails.
            max_tries (int, optional): Maximum number of tries to generate a new word. Defaults to -1.
            blacklist_fn (Optional[Callable[[str], bool]], optional): Function to check if the new word is blacklisted. Defaults to None.

        Returns:
            str or None: New word based on the definition of the input word. If the word does not exist, it returns None.
        """
        word_exists = True
        n_tries = 0
        while word_exists and (n_tries < max_tries or max_tries == -1):
            if n_tries > 0:
                self.tr.seed = self.seed + n_tries

            new_word = self.tr(word)
            new_word = safety_check(new_word)
            # Check in the blacklist
            if not blacklist_fn is None:
                word_exists = False or blacklist_fn(new_word)
            else:
                word_exists = False
            n_tries += 1

            if word_exists:
                print(f"[WARN] Word '{new_word}' from '{word}' already exists in the KG. Trying again... (try {n_tries}/{max_tries})")
        # Restore seed
        self.tr.seed = self.seed
        if not word_exists:
            return new_word
        else:
            return None