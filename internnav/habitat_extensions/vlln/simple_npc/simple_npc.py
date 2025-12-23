import base64
import random

from .prompt import DISAMBIGUATION_PROMPT, TEMPLATE


class SimpleNPC:
    """
    A simple LLM-driven NPC for language-based navigation interaction.

    Args:
        max_interaction_turn (int): Maximum number of retry attempts when querying the language model.
        model_name (str): Name of the LLM model to use.
        openai_api_key (str): Path to a text file containing the OpenAI API key.
        base_url (Optional[str]): Optional base URL for OpenAI-compatible APIs.
    """

    def __init__(
        self,
        max_interaction_turn: int,
        model_name: str,
        openai_api_key: str,
        base_url: str = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            raise ImportError('ModuleNotFoundError: No module named \'openai\'. Please install it first.')

        self.model_name = model_name
        self.max_turn = max_interaction_turn
        self.history_messages = []
        with open(openai_api_key, 'r', encoding='utf-8') as file:
            openai_api_key = file.read().strip()
        try:
            self.llm = OpenAI(api_key=openai_api_key, base_url=base_url)
        except Exception as e:
            print(f'Failed to initialize OpenAI: {e}')

    def get_room_name(self, room):
        room_name_dict = {
            "living region": "living room",
            "stair region": "stairs",
            "bathing region": "bathroom",
            "storage region": "storage room",
            "study region": "study room",
            "cooking region": "kitchen",
            "sports region": "sports room",
            "corridor region": "corridor",
            "toliet region": "toilet",
            "dinning region": "dining room",
            "resting region": "resting room",
            "open area region": "open area",
            "other region": "area",
        }
        return room_name_dict[room]

    def answer_question(
        self, question: str, instance_id: str, object_dict: dict, task_done: bool, path_description: str, mode: str
    ):
        if mode == 'one_turn':
            goal_information = ''
            goal_information += 'room: ' + self.get_room_name(object_dict[instance_id]['room']) + '\n'
            goal_information += '\n'.join(
                [
                    f'{a.lower()}: {i.lower()}'
                    for a, i in object_dict[instance_id]['unique_description'].items()
                    if a in ['color', 'texture', 'material', 'shape', 'placement'] and len(i) > 0
                ]
            )
            nearby_objects = [
                object_dict[obj]['unique_description']['fine grained category'].lower()
                for obj, _ in object_dict[instance_id]['nearby_objects'].items()
                if obj in object_dict and isinstance(object_dict[obj]['unique_description'], dict)
            ]
            if len(nearby_objects) > 0:
                goal_information += '\nnearby objects: ' + ','.join(nearby_objects)
            goal_information += 'whole description: ' + object_dict[instance_id]['caption']
            answer = self.ask_directly(
                template_type="one_turn_prompt",
                question=question,
                goal_information=goal_information,
                path_description=path_description,
                task_done=task_done,
            )
            return answer
        elif mode == 'two_turn':
            answer = self.ask_directly(
                template_type="two_turn_prompt_0",
                question=question,
            )
            if 'path' in answer.lower():
                return path_description
            elif 'disambiguation' in answer.lower():
                if task_done:
                    return random.choice(DISAMBIGUATION_PROMPT['yes'])
                else:
                    return random.choice(DISAMBIGUATION_PROMPT['no'])
            elif 'information' in answer.lower():
                goal_information = ''
                goal_information += 'room: ' + self.get_room_name(object_dict[instance_id]['room']) + '\n'
                goal_information += '\n'.join(
                    [
                        f'{a.lower()}: {i.lower()}'
                        for a, i in object_dict[instance_id]['unique_description'].items()
                        if a in ['color', 'texture', 'material', 'shape', 'placement'] and len(i) > 0
                    ]
                )
                nearby_objects = [
                    object_dict[obj]['unique_description']['fine grained category'].lower()
                    for obj, _ in object_dict[instance_id]['nearby_objects'].items()
                    if obj in object_dict and isinstance(object_dict[obj]['unique_description'], dict)
                ]
                if len(nearby_objects) > 0:
                    goal_information += '\nnearby objects: ' + ','.join(nearby_objects)
                goal_information += 'whole description: ' + object_dict[instance_id]['caption']
                answer = self.ask_directly(
                    template_type="one_turn_prompt",
                    question=question,
                    goal_information=goal_information,
                    path_description=path_description,
                    task_done=task_done,
                )
                answer = self.answer_question(question, instance_id, object_dict, task_done, answer, 'one_turn')
            return answer
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def ask_directly(self, template_type, **kwargs):
        def generate_prompt(template_type, **kwargs):
            prompt = TEMPLATE.get(template_type, None)
            if prompt is None:
                raise ValueError(f"Template type '{template_type}' not found.")
            prompt = prompt.format(**kwargs)
            return prompt

        messages = []
        image_bufs = kwargs.get('images', None)
        cnt = 0
        prompt = generate_prompt(template_type, **kwargs)
        content = [{'type': 'text', 'text': prompt}]
        if image_bufs is not None:
            for im_id, image_buf in enumerate(image_bufs):
                img_encoded = base64.b64encode(image_buf.getvalue()).decode('utf-8')
                image_buf.close()
                item = {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{img_encoded}',
                        'detail': 'high',
                    },
                    'index': im_id,
                }
                content.append(item)
        messages.append({'role': 'user', 'content': content})

        while cnt < self.max_turn:
            try:
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                result = response.choices[0].message.content
                break
            except Exception as e:
                print(e)
                cnt += 1
                result = None
        return result
