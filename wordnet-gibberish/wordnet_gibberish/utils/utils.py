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


from typing import Optional
import wordfreq

def isVowel(ch : str) -> bool:
	"""Check if a character is a vowel.

	Args:
		ch (str): Character.
		
	Returns:
		bool: True if the character is a vowel, False otherwise.
	"""
	return (0x208222 >> (ord(ch) & 0x1f)) & 1
	# same as (2130466 >> (ord(ch) & 31)) & 1; 

def shorten_uri(concept : str) -> str:
	"""Shorten the URI of a concept.

	Args:
		concept (str): Concept URI.

	Returns:
		str: Shortened concept URI.
	"""
	return f"id:{concept.split('/id/')[-1][:-1]}"

def get_top_words(k : int = 2000, banlist : Optional[list] = None) -> set[str]:
	"""Get the top k most common English words.

	Args:
		k (int, optional): Number of top words to get. Defaults to 500.
		banlist (Optional[list], optional): List of words to exclude. Defaults to None.

	Returns:
		set[str]: List of top words (not ordered).
	"""
	assert k > 0, "k must be greater than 0"
	top_words = set(wordfreq.top_n_list("en", k))
	if banlist is not None:
		banlist_ = set(banlist)
		top_words = top_words.difference(banlist_)
	return top_words

def safety_check(word : str) -> str:
	"""Check if a word is safe to use.

	Args:
		word (str): Word to check.

	Returns:
		str: Word if it's safe, an empty string otherwise.
	"""
	# Remove trailing punctuation and lowercase
	new_word = word
	new_word = new_word.strip().lower()
	forbidden_trailing_characters = set([".", ",", ";", ":", "!", "?", "-", " "])
	forbidden_characters = set(["'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "«", "»", "‘", "’", "“", "”", "„", "‟", "‹", "›", "‛", "❛", "❜", "❝", "❞", "❮", "❯", "❨", "❩", "❪", "❫",])
	split_patterns = set([
		", ", "<|bd|>", ",", ".", "  "
	])
	# Remove all forbidden characters
	for char in forbidden_characters:
		new_word = new_word.replace(char, "")
		if new_word == "":
			return ""

	# Take care of special symbols too
	while new_word[-1] in forbidden_trailing_characters:
		new_word = new_word[:-1]
	while new_word[0] in forbidden_trailing_characters:
		new_word = new_word[1:]

	# If there is a split to do, attach all the words together
	for pattern in split_patterns:
		if pattern in new_word:
			new_word = new_word.replace(pattern, "")

	return new_word


def escape_unicode(input_string):
    escaped_string = ""
    for char in input_string:
        # Check if the character falls within the Latin-1 Supplement Unicode block
        if '\u0080' <= char <= '\u00FF':
            # Get the Unicode code point of the character
            code_point = ord(char)
            # Format the code point as a 4-digit hexadecimal string
            escaped_char = f"\\u{code_point:04x}"
            escaped_string += escaped_char
        else:
            # Leave non-special characters unchanged
            escaped_string += char
    return escaped_string