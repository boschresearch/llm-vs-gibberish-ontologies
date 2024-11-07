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


# Import ABC and abstractmethod from abc
from abc import ABC, abstractmethod
from typing import Optional, Callable

# Create the Translator class
class Translator(ABC):
    def __init__(self):
        pass

    def translate(self, word: str, definition: str, pos: str,
                  do_sample: bool = False,
                  max_tries : int = -1,
                  blacklist_fn: Optional[Callable[[str], bool]] = None) -> str:
        pass

    def __call__(self, word: str, definition: str, pos: str,
                  do_sample: bool = False,
                  max_tries: int = -1,
                  blacklist_fn: Optional[Callable[[str], bool]] = None) -> str:
        return self.translate(word, definition, pos, do_sample, max_tries, blacklist_fn)