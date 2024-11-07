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
import requests
from typing import Optional, Callable
from time import sleep

class GPT2Translator(Translator):
    def __init__(self,
                port: int = 8000,
                host: str = "localhost",
                timeout: int = 30,
                temperature: Optional[float] = None,
                ):
        self.port = port
        self.host = host
        self.timeout = timeout
        self.temperature = temperature
        self.url = f"http://{self.host}:{self.port}/word_from_definition"

    def make_request_to_gpt2(self, definition: str,
                            do_sample: Optional[bool] = None,
                            temperature: Optional[float] = None
                            ) -> str:
        # The input of the POST request must be a dictionary
        # {
        # "definition": "string",
        # "do_sample": false,
        # "temperature": 0,
        # "dataset": {
        #     "name": "OED"
        # }
        # }
        if do_sample is None:
            do_sample = False

        data = {
            "definition": definition,
            "do_sample": do_sample,
            "temperature": temperature,
            "dataset": {
                "name": "OED"
            }
        }
        response = requests.post(self.url, json=data, timeout=self.timeout)
        return response.json()["word"]
    

    def translate(self, word: str, definition: str, pos: str,
                  do_sample: bool = False,
                  max_tries: int = -1,
                  blacklist_fn: Optional[Callable[[str], bool]] = None
                  ) -> str:
        """Make a new nonexistent word based on the definition of this word using GPT2.

        Args:
            word (str): Word to translate. This is not used in this translator.
            definition (str): Definition of the word to translate.
            pos (str): Part of speech of the word to translate (e.g. "noun", "verb")
            do_sample (bool, optional): Whether to use sampling to generate the new word. Defaults to False.
                            Note that if max_tries is greater than 1, this will be set to True after the first try fails.
            max_tries (int, optional): Maximum number of tries to generate a new word. Defaults to -1.
            blacklist_fn (Optional[Callable[[str], bool]], optional): Function to check if the new word is blacklisted. Defaults to None.

        Returns:
            str or None: New word based on the definition of the input word. If the word does not exist, it returns None.
        """
        word_exists = True
        n_tries = 0
        temperature = self.temperature
        definition_ = f"({pos}) {definition}"

        while word_exists and (n_tries < max_tries or max_tries == -1):
            # Increase the temperature if the previous attempt did not return a new word
            if temperature is None and n_tries > 0:
                temperature = 0.1
            else:
                temperature = min(temperature + 0.1, 0.95)
            
            if n_tries > 0 and not do_sample:
                if pos == "noun":
                    definition_ = f"(noun) a {definition}, that doesn't exist"
                elif pos == "verb":
                    definition_ = f"(verb) to {definition}, in a way that doesn't exist"
                elif pos == "adverb":
                    definition_ = f"(adverb) in a {definition} way that doesn't exist"
                else:
                    # adjective
                    definition_ = f"(adjective) {definition} in a way that doesn't exist"
            
            try:
                new_word = self.make_request_to_gpt2(definition_,
                                                    do_sample = (do_sample or (n_tries > 0)),
                                                    temperature=temperature)
                word_exists = new_word["probablyExists"]
                new_word_ = new_word["word"]
                new_word_ = safety_check(new_word_)
                if not blacklist_fn is None:
                    word_exists = word_exists or blacklist_fn(new_word_)
            except requests.exceptions.Timeout:
                print(f"[WARN] Timeout while trying to generate a new word. Trying again... (try {n_tries}/{max_tries})")
                word_exists = True
                new_word_ = None
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] {e}")
                word_exists = True
                new_word_ = None

            n_tries += 1
            if word_exists:
                sleep(2)
        
        if not word_exists:
            return new_word_
        else:
            return None





