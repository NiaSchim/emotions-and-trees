# agi.py
import llamacpp
import json
import tracery
import text2emotion as te
import stanfordnlp
from random import choice
import random
import time
import os
from tracery.modifiers import base_english
import csv
import pandas as pd
EMOTION_INERTIA = 0.38095238095
PKS_LEAK_RATE = 0.9999
DEFAULT_LEAK_RATE = 0.99995
WEIGHT_THRESHOLD = 0.0001
nlp = stanfordnlp.Pipeline()

class AGI:
    def __init__(self, model_path, emotional_decoder_csv_path):
        self.last_user_input = "None"
        self.model_path = model_path
        self.model = self.initialize_model()
        self.emotional_decoder_csv_path = "emotional_decoder.csv"
        self.nlp = stanfordnlp.Pipeline()
        self.current_emotion = None
        self.uksekspks = self.initialize_pks()
        self.override_generate_response = False
        self.user_target_emotion = None
        self.bot_target_emotion = self.get_emotion_coordinates("happy")
        self.target_emotion = self.calculate_target_emotion(self.user_target_emotion, self.bot_target_emotion)
        self.questions1 = [            "how comfortable are your emotions on a scale of 0 to 4",            "how familiar are your emotions on a scale of 0 to 4",            "how changing novel or dynamic are your emotions on a scale of 0 to 4",            "how uncomfortable are your emotions on a scale of 0 to 4",            "how comfortable is your body on a scale of 0 to 4",            "how familiar are your bodily sensations on a scale of 0 to 4",            "how changing novel or dynamic are your bodily sensations on a scale of 0 to 4 ",            "how uncomfortable is your body on a scale of 0 to 4",            "how comfortable is your current environment on a scale of 0 to 4",            "how familiar is your environment on a scale of 0 to 4",            "how changing novel or dynamic is your environment on a scale of 0 to 4",            "how uncomfortable is your environment on a scale of 0 to 4",            "how comfortable are your intentions on a scale of 0 to 4",            "how familiar or familiarity-seeking are your intentions on a scale of 0 to 4 ",            "how changing novel or dynamic are your intentions on a scale of 0 to 4",            "how uncomfortable are your intentions on a scale of 0 to 4",            "note: 'the other' could be any person place or thing that has your attention right now. How comfortable does the other make you on a scale of 0 to 4",            "how familiar is the other on a scale of 0 to 4",            "how changing novel or dynamic is the other on a scale of 0 to 4  ",            "how uncomfortable does the other make you on a scale of 0 to 4",            "how comfortable is your social group on a scale of 0 to 4",            "how familiar is your social group on a scale of 0 to 4",            "how changing novel or dynamic is your social group on a scale of 0 to 4     ",            "how uncomfortable is your social group on a scale of 0 to 4",            "write '1' if you are focused on maintaining or changing your emotions, or else write '0'.",            "write '1' if you are focused on maintaining or changing how your body feels, or else write '0'.",            "write '1' if you are focused on maintaining or changing your environment, or else write '0'.",            "write '1' if you are focused on maintaining or changing your intentions, or else write '0'.",            "write '1' if you are focused on maintaining or changing The Other, or else write '0'.",            "write '1' if you are focused on maintaining or changing your social place, or else write '0'."        ]
        self.questions2 = [    "how comfortable do you want your emotions to be on a scale of 0 to 4",    "how familiar do you want your emotions tobe on a scale of 0 to 4",    "how changing novel or dynamic do you want do you want your  emotions to be on a scale of 0 to 4",    "how comfortable is do you want your  body to be on a scale of 0 to 4",    "how familiar to be do you want your  bodily sensations to be on a scale of 0 to 4",    "how changing novel or dynamic to be do you want your  bodily sensations to be on a scale of 0 to 4 ",    "how comfortable do you want your  current environment to be on a scale of 0 to 4",    "how familiar do you want your  environment to be on a scale of 0 to 4",    "how changing novel or dynamic do you want your  environment to be on a scale of 0 to 4  ",    "how comfortable do you want your  intentions to be on a scale of 0 to 4",    "how familiar or familiarity-seeking do you want your  intentions to be on a scale of 0 to 4 ",    "how changing novel or dynamic do you want your  intentions to be on a scale of 0 to 4",    "note: 'the other' could be any person place or thing that has do you want your  attention right now.",    "how comfortable do you want the other make you on a scale of 0 to 4",    "how familiar do you want the other to be on a scale of 0 to 4",    "how changing novel or dynamic do you want other to be on a scale of 0 to 4  ",    "how comfortable do you want your  social group to be on a scale of 0 to 4",    "how familiar do you want your  social group to be on a scale of 0 to 4",    "how changing novel or dynamic do you want your  social group to be on a scale of 0 to 4     ",    "write '1' if you want to be focused to be on maintaining or changing your  emotions, or else write '0'.",    "write '1' if you want to be focused to be on maintaining or changing how your  body feels, or else write '0'.",    "write '1' if you want to be focused to be on maintaining or changing your  environment, or else write '0'.",    "write '1' if you want to be focused on maintaining or changing your  intentions, or else write '0'.",    "write '1' if you want to be focused on maintaining or changing your social place, or else write '0'."]
        self.message_count = 0
        self.current_question_index_number = 0  # Move this line here
        self.is_waiting_for_valid_answer = False
        survey1_results = self.get_results_of_questions(self.questions1)
        survey2_results = self.get_results_of_questions(self.questions2)
        self.current_emotion = self.get_emotion_coordinates(survey1_results)
        self.user_target_emotion = self.get_emotion_coordinates(survey2_results)


    def initialize_pks(self):
        try:
            with open("uksekspks.json", "r") as f:
                pks_data = json.load(f)
                pks = UKSEKSPKS("uksekspks.json")
                pks.data = pks_data
        except FileNotFoundError:
            initial_coords = pd.read_csv(self.emotional_decoder_csv_path, index_col=0).iloc[0]
            pks = UKSEKSPKS("uksekspks.json")
            pks.add_profile("default", initial_coords)
            with open("uksekspks.json", "w") as f:
                json.dump(pks.to_dict(), f, indent=4)
        return pks

    def progress_callback(progress):
        print("Progress: {:.2f}%".format(progress * 100))
        sys.stdout.flush()


    def initialize_model(self):
        params = llamacpp.InferenceParams.default_with_callback(self.progress_callback)
        params.path_model = self.model_path
        params.seed = random.randint(10000, 99999)
        params.repeat_penalty = 1.0
        model = llamacpp.LlamaInference(params)
        return model

    def generate_suggestion(self, prompt):
        prompt_tokens = self.model.tokenize(prompt, True)
        self.model.update_input(prompt_tokens)
        self.model.ingest_all_pending_input()

        suggestion = ""
        while True:
            self.model.eval()
            token = self.model.sample()
            text = self.model.token_to_str(token)
            if text == "\n":
                break
            suggestion += text

        return suggestion.strip()

    def follows_logically(self, sequence):
        sentence = ' '.join(sequence)
        try:
            doc = self.nlp(sentence)
            return True
        except:
            return False

    def parse_emotional_decoder_csv(self):
        emotions_dict = {}
        with open(self.emotional_decoder_csv_path, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                emotions_dict[row[0]] = [float(val) for val in row[1:]]
        return emotions_dict

    def get_emotion_coordinates(self, survey_results):
        if isinstance(survey_results, dict):
            coordinates = [value for key, value in survey_results.items() if key != "name"]
            return coordinates
        else:
            emotion_coordinates = self.parse_emotional_decoder_csv()
            if survey_results in emotion_coordinates:
                coordinates_str = emotion_coordinates[survey_results]
                if isinstance(coordinates_str, list):
                    coordinates = [float(c) for c in coordinates_str[1:]]  # Convert to floats and skip the name part
                else:
                    coordinates = [float(c) for c in coordinates_str.split(',')[1:]]  # Convert to floats and skip the name part
                return coordinates
            return None

    def average_emotional_coordinates(self, coord1, coord2):
        return [(a + b) / 2 for a, b in zip(coord1, coord2)]

    def calculate_target_emotion(self, user_target_emotion, bot_target_emotion):
        if user_target_emotion is None:
            return bot_target_emotion
        else:
            return self.average_emotional_coordinates(user_target_emotion, bot_target_emotion)

    def update_user_emotion(self, user_emotion, new_emotion, survey_results):
        if new_emotion not in self.uksekspks.data["emotions"]:
            new_emotion_name = self.handle_new_emotion_name(new_emotion, survey_results)
            self.uksekspks.data["emotions"][new_emotion_name] = user_emotion
        else:
            self.uksekspks.data["emotions"][new_emotion] = user_emotion
        self.uksekspks.save_data()

    def get_last_user_input(self):
        return self.last_user_input

    def generate_response_with_target_emotion(self, input_text):
        self.last_user_input = input_text
        response = self.generate_suggestion(input_text)
        user_emotion = self.identify_emotion(input_text)
        response_emotion = self.identify_emotion(response)

        self.update_emotional_synapse_weights(input_text, response)
        user_emotion_label = max(user_emotion, key=user_emotion.get) # get the label with the highest probability
        eks_emotion = self.uksekspks.generate_eks_sentence("#" + user_emotion_label + "#")
        text2emotion_emotion = te.get_emotion(input_text)
        predicted_emotion = self.calculate_predicted_emotion(eks_emotion, text2emotion_emotion)
        self.current_emotion = self.calculate_new_current_emotion(self.current_emotion, predicted_emotion)

        if response_emotion != target_emotion:
            emotion_coordinates = self.parse_emotional_decoder_csv()
            target_coordinates = emotion_coordinates.get(target_emotion)
            user_coordinates_str = emotion_coordinates.get(user_emotion_label) # use the extracted label instead of the dictionary
            if isinstance(response_emotion, dict):
                response_emotion_key = next(iter(response_emotion))
                response_coordinates_str = emotion_coordinates.get(response_emotion_key)
            else:
                response_coordinates_str = emotion_coordinates.get(response_emotion)

            if target_coordinates and user_coordinates_str and response_coordinates_str:
                user_coordinates = [int(c) for c in user_coordinates_str.split(',')] # convert to integers
                response_coordinates = [int(c) for c in response_coordinates_str.split(',')] # convert to integers
                avg_coordinates = self.average_emotional_coordinates(user_coordinates, response_coordinates)
                for emotion, coordinates_str in emotion_coordinates.items():
                    coordinates = [int(c) for c in coordinates_str.split(',')] # convert to integers
                    if coordinates == avg_coordinates:
                        target_emotion = emotion
                        break

                response = self.uksekspks.generate_eks_sentence("#" + target_emotion + "#")

        return response

    def handle_new_emotion_name(self, new_emotion, survey_results):
        emotion_coordinates = self.parse_emotional_decoder_csv()

        # Extract the 30 integers from the dictionary, ignoring the arbitrary name key
        survey_results = next(value for key, value in survey_results_dict.items() if key != "arbitrary_name")

        # Check if emotion already exists in the emotional decoder
        if new_emotion in emotion_coordinates:
            existing_coordinates = emotion_coordinates[new_emotion]
            if existing_coordinates == survey_results:
                # Strengthen synapses of existing emotion with corresponding word trees
                self.uksekspks.strengthen_word_trees(existing_coordinates, survey_results)
                return new_emotion
            else:
                # If the emotion already exists with different coordinates, rename the existing emotion
                i = 1
                while True:
                    old_emotion = f"{new_emotion} (old definition {i})"
                    if old_emotion not in emotion_coordinates:
                        break
                    i += 1
                emotion_coordinates[old_emotion] = emotion_coordinates.pop(new_emotion)
                new_emotion = old_emotion

        # Add the new emotion to the emotional decoder
        emotion_coordinates[new_emotion] = survey_results
        self.save_emotional_decoder_csv(emotion_coordinates)

        # Strengthen synapses of new emotion with corresponding word trees
        self.uksekspks.strengthen_word_trees(survey_results, survey_results)

        return new_emotion

    def wait_and_get_user_input(self):
        # Wait for user to input something
        user_input = None
        while not user_input:
            time.sleep(0.1)
            user_input = self.get_last_user_input()
        # Clear last user input
        self.clear_last_user_input()
        return user_input

    def override_output(self, output):
        # Override the output of generate_response method
        self.override_generate_response = output

    def ask_question(self, question):
        question = self.questions1[int(self.current_question_index_number)] if int(self.current_question_index_number) < len(self.questions1) else self.questions2[int(self.current_question_index_number) - len(self.questions1)]
        self.override_output(question)

    def get_results_of_questions(self, questions):
        results = []
        for question in questions:
            self.ask_question(question)
            answer = self.validate_answer(self.get_last_user_input(),self.current_question_index_number,self.get_current_survey_number())
            results.append(answer)
        survey_emotional_coordinate = {'name': 'survey', **{f'coord{i}': int(results[i]) for i in range(len(results))}}
        return survey_emotional_coordinate
        self.survey1_results = self.get_results_of_questions(self.questions1)
        self.survey2_results = self.get_results_of_questions(self.questions2)


    def start_surveys(self):
        self.current_question_cycle = cycle(self.questions1 + self.questions2)
        self.current_survey_results = get_results_of_questions(questions).survey_emotional_coordinate

    def validate_answer(self, answer, index, survey_number):
        if not answer.isdigit():
            return False

        answer_int = int(answer)

        if survey_number == 1:
            if 0 <= index < 24:  # First 24 questions
                return 0 <= answer_int <= 4
            elif 24 <= index < 30:  # Last 6 questions
                return answer_int in (0, 1)
            else:
                return False
        elif survey_number == 2:
            if 0 <= index < 18:  # First 18 questions
                return 0 <= answer_int <= 4
            elif 18 <= index < 24:  # Last 6 questions
                return answer_int in (0, 1)
            elif 24 <= index:  # Last 6 questions
                self.user_target_emotion = self.get_emotion_coordinates(survey2_results)
                self.target_emotion = self.calculate_target_emotion(self.user_target_emotion, self.bot_target_emotion)
            else:
                return False
        else:
            return False

    def get_current_survey_number(self):
        if self.current_question_index_number < len(self.questions1):
            return 1
        else:
            self.current_emotion = self.get_emotion_coordinates(survey1_results)
            return 2

    def handle_survey_results(self):
        survey1_results = self.current_survey_results[:len(self.questions1)]
        survey2_results = self.current_survey_results[len(self.questions1):]

        self.current_emotion = self.get_emotion_coordinates(survey1_results)
        self.user_target_emotion = self.get_emotion_coordinates(survey2_results)
        self.target_emotion = self.calculate_target_emotion(self.user_target_emotion, self.bot_target_emotion)

    def process_survey_answer(self, input_text):
        if self.validate_answer(input_text, self.current_question_index_number, self.get_current_survey_number()):
            self.current_survey_results = int(input_text)
            self.current_question_index_number += 1
            if self.current_question_index_number >= len(self.current_survey_results):
                self.handle_survey_results()
                self.current_question_index_number = 0
                self.is_waiting_for_valid_answer = False
        else:
            self.override_output("Invalid input. Please enter a valid answer.")

    def update_surveys(self):
        self.message_count += 1
        survey2_zero_cycle = 6
        survey2_zero_count = 0
        survey2_question_count = 0

        if self.message_count % 23 == 0:
            self.is_waiting_for_valid_answer = True
            self.ask_question(self.current_question_index_number)

        if self.message_count % 51 == 0:
            self.is_waiting_for_valid_answer = True
            self.ask_question(self.current_question_index_number + len(self.questions1))

            # Add zeros as required
            if survey2_question_count < survey2_zero_cycle * 3:
                survey2_zero_count += 1
                if survey2_zero_count == 3:
                    self.current_survey_results.insert(self.current_question_index_number + len(self.questions1) + survey2_question_count, 0)
                    survey2_zero_count = 0
                    survey2_question_count += 1

    def generate_response(self, input_text):
        if self.is_waiting_for_valid_answer:
            self.process_survey_answer(input_text)
            if self.is_waiting_for_valid_answer:
                self.ask_question(self.current_question_index_number)
            else:
                self.update_surveys()
                return self.generate_response(input_text)
        else:
            if self.override_generate_response:
                output = self.override_generate_response
                self.override_generate_response = None
                self.update_surveys()
            else:
                output = self.generate_response_with_target_emotion(input_text)
                self.update_surveys()
            return output

    def identify_emotion(self, text):
        return te.get_emotion(text)

    def calculate_predicted_emotion(self, eks_emotion, t2e_emotion):
        eks_strength = self.uksekspks.get_synapse_strengths("EKS")
        t2e_strength = self.uksekspks.get_synapse_strengths("T2E")

        eks_w = sum(eks_strength.values())
        t2e_w = sum(t2e_strength.values())

        if eks_w > t2e_w:
            return eks_emotion
        elif t2e_w > eks_w:
            return t2e_emotion
        else:
            return choice([eks_emotion, t2e_emotion])

    def calculate_new_current_emotion(self, last_emotion, predicted_emotion):
        if not last_emotion:
            return predicted_emotion
        else:
            return [0.8 * last_emotion[i] + 0.2 * predicted_emotion[i] for i in range(len(last_emotion))]

    def update_emotional_synapse_weights(self, prompt, response):
        prompt_words = prompt.split()
        response_words = response.split()

        # Update weights for prompt and response
        self.uksekspks.update_tree(prompt_words, self.current_emotion, is_user_input=True)
        self.uksekspks.update_tree(response_words, self.current_emotion, is_user_input=False)


class UKSEKSPKS:
    def __init__(self, uksekspks_json_path):
        self.uksekspks_json_path = uksekspks_json_path
        self.load_data()

    def load_data(self):
        if os.path.exists(self.uksekspks_json_path):
            with open(self.uksekspks_json_path, "r") as file:
                self.data = json.load(file)
        else:
            self.data = {
                "UKS": {},
                "EKS": {},
                "PKS": {},
                "weights": {}
            }
            self.save_data()

    def save_data(self):
        with open(self.uksekspks_json_path, "w") as file:
            json.dump(self.data, file, indent=4)

    def get_weight_keys(self, depth, word):
        """Return the weight keys for a given depth and word."""
        return [f"UK_{depth}_{word}", f"SK_{depth}_{word}", f"PKS_{depth}_{word}"]

    def generate_uks_sentence(self, user_input):
        uks_grammar = tracery.Grammar(self.data["UKS"])
        uks_grammar.add_modifiers(tracery.modifiers.base_english)
        return uks_grammar.flatten(user_input)

    def generate_eks_sentence(self, user_input):
        eks_grammar = tracery.Grammar(self.data["EKS"])
        eks_grammar.add_modifiers(tracery.modifiers.base_english)
        return eks_grammar.flatten(user_input)

    def generate_pks_sentence(self, user_input):
        pks_grammar = tracery.Grammar(self.data["PKS"])
        pks_grammar.add_modifiers(tracery.modifiers.base_english)
        return pks_grammar.flatten(user_input)

    def update_weights(self, encountered_weights):
        for weight_key in self.data["weights"]:
            if weight_key not in encountered_weights:
                self.data["weights"][weight_key] *= 0.99995
                if self.data["weights"][weight_key] < 0.0001:
                    self.data["weights"][weight_key] = 0
        self.save_data()

    def predict_current_emotion(self, input_words):
        eks_emotion = self.get_eks_emotion(input_words)
        t2e_emotion = self.get_t2e_emotion(input_words)
        eks_synapses = self.get_synapse_strengths("EKS", input_words)
        t2e_synapses = self.get_synapse_strengths("T2E", input_words)

        if all(s > 0.54 for s in eks_synapses.values()):
            return eks_emotion
        else:
            return self.average_emotions(eks_emotion, t2e_emotion)

    def update_tree(self, input_words, current_emotion, is_user_input=True):
        """Update the tree and weights for the given input words and current emotion."""

        # Check if the current emotion exists in the data dictionary, and add it with an empty dictionary if not
        if current_emotion not in self.data:
            self.data[current_emotion] = {}

        # Update weights for all encountered keys
        encountered_weights = set()
        for depth, word in enumerate(input_words):
            for weight_key in self.get_weight_keys(depth, word):
                encountered_weights.add(weight_key)

                # Check if the weight key exists in the weights dictionary, and add it with a default value if not
                if weight_key not in self.data["weights"]:
                    self.data["weights"][weight_key] = 1.0

                # Update the weight according to the input type and leak rate
                if is_user_input and weight_key.startswith("PKS"):
                    self.data["weights"][weight_key] *= PKS_LEAK_RATE
                else:
                    self.data["weights"][weight_key] *= DEFAULT_LEAK_RATE

                # Remove the weight key if it falls below the threshold
                if self.data["weights"][weight_key] < WEIGHT_THRESHOLD:
                    del self.data["weights"][weight_key]

                # Logical assessment: update weights based on whether they follow logically or not
                if depth == -1:
                    self.data["weights"][weight_key] *= 1
                else:
                    depth += 1
                    if not self.follows_logically(input_words[:depth]):
                        self.data["weights"][weight_key] *= (1 - (1 / (3 * 2 * depth)))
                    else:
                        self.data["weights"][weight_key] *= (1 + (2 / (3 * 2 * depth)))

                # Boost synapse strength between word trees and emotional coordinates that are already connected
                if current_emotion and emotion_pair in self.data[weight_key]:
                    self.data[weight_key][emotion_pair] *= EMOTION_INERTIA

                # Check if the tree key exists in the data dictionary, and add it with an empty dictionary if not
                for key in [weight_key + '_prompt', weight_key + '_response']:
                    tree_key = key.split("_", 1)[0]
                    if tree_key not in self.data:
                        self.data[tree_key] = {}

                    current_node = self.data[tree_key]
                    if word not in current_node:
                        current_node[word] = {}

                    current_node = current_node[word]

        # Update weights for all non-encountered keys
        self.update_weights(encountered_weights)

    def follows_logically(self, sequence):
        # Implement the logic assessment here, returning True if the sequence is deemed logical, and False otherwise.
        # For now, we will simply return True. You should replace this with your own logic assessment function.
        return True

    def get_synapse_strengths(self, tree_key):
        synapse_strengths = {}
        for weight_key, weight_value in self.data["weights"].items():
            if weight_key.startswith(tree_key):
                for emotion_key, emotion_value in self.data[weight_key].items():
                    if emotion_key not in synapse_strengths:
                        synapse_strengths[emotion_key] = 0
                    synapse_strengths[emotion_key] += weight_value * emotion_value
        return synapse_strengths
