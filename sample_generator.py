"""
A module that generates samples imitating game moments.

"""
import os
import random
from typing import Optional, Tuple
import pickle
import numpy as np
import cv2

class SampleGenerator():

    def __init__(self) -> None:

        self.scale_dict: dict = {
            "battleground": 1,
            "platform": 0.52,
            "enemy": 0.5,
            "cuphead": 0.54
        }

        self.project_dir: str = os.path.dirname(os.path.abspath(__file__))

        self.training_set_dir: str = os.path.join(self.project_dir, "Training set")
        self.battleground_dir: str = os.path.join(self.project_dir, "Image source", "Floral_fury_battleground_cut.png")
        self.platform_dir: str = os.path.join(self.project_dir, "Image source", "Cagney Carnation", "Platform")
        self.enemy_dir: str = os.path.join(self.project_dir, "Image source", "Cagney Carnation", "Enemy")
        self.objects_dir: str = os.path.join(self.project_dir, "Image source", "Cagney Carnation", "Objects")
        self.cuphead_dir: str = os.path.join(self.project_dir, "Image source", "Cuphead")
        
        self.platform_sprints: list = list()
        self.enemy_sprints: list = list()
        self.objects_sprints: dict = dict()
        self.cuphead_sprints: list = list()

        for (dirpath, dirnames, filenames) in os.walk(self.platform_dir):
            self.platform_sprints += [os.path.join(dirpath, file) for file in filenames]

        for (dirpath, dirnames, filenames) in os.walk(self.enemy_dir):
            self.enemy_sprints += [os.path.join(dirpath, file) for file in filenames]

        for (dirpath, dirnames, filenames) in os.walk(self.objects_dir):
            self.objects_sprints[os.path.basename(dirpath)] = [os.path.join(dirpath, file) for file in filenames]

        for (dirpath, dirnames, filenames) in os.walk(self.cuphead_dir):
            self.cuphead_sprints += [os.path.join(dirpath, file) for file in filenames]
        

    def read_image(self, img_dir: str, scale_key: str, horizontal_flip=False) -> cv2.typing.MatLike:
        """Reading image method with rescaling and fliping objects.

        Args:
            img_dir (str): image directory.
            scale_key (str): scaling factor.
            horizontal_flip (bool, optional): flipping flag. Defaults to False.

        Raises:
            FileNotFoundError: directory img_dir not found.

        Returns:
            cv2.typing.MatLike: read image.
        """
        image: Optional[cv2.typing.MatLike] = cv2.imread(filename=img_dir, flags=cv2.IMREAD_UNCHANGED)

        if image is None:
            raise FileNotFoundError(img_dir)

        image = cv2.resize(src=image, dsize=None, fx=self.scale_dict.get(scale_key, 0.5), fy=self.scale_dict.get(scale_key, 0.5))

        if horizontal_flip:
            image = cv2.flip(image, 1)

        return image
    

    def paste_image(self, background_img: cv2.typing.MatLike, object_img: cv2.typing.MatLike, x_tl: int, y_tl: int) -> cv2.typing.MatLike:
        """Pasting object on background method that cuts exceeding pixels from object.

        Args:
            background_img (cv2.typing.MatLike): background image.
            object_img (cv2.typing.MatLike): object image to paste.
            x_tl (int): top left x coordinate to place object.
            y_tl (int): top left y coordinate to place object.

        Raises:
            Exception: object is bigger than background image.
            Exception: used x_tl exceeds boundaries of the background.
            Exception: used y_tl exceeds boundaries of the background.

        Returns:
            cv2.typing.MatLike: background with pasted object.
        """
        h_bg, w_bg = background_img.shape[:2]
        h_obj, w_obj = object_img.shape[:2]

        if h_obj > h_bg or w_obj > w_bg:
            raise Exception(f"The object exceeds the dimensions of the background. h_obj <= h_bg ({h_obj} <= {h_bg}), w_obj <= w_bg ({w_obj} <= {w_bg})")
        
        if x_tl > w_bg or x_tl + w_obj < 0:
            raise Exception(f"The object exceeds the boundaries of the background. x_tl <= w_bg ({x_tl} <= {w_bg}), x_tl + w_obj >= 0 ({x_tl + w_obj} >= 0)") 
        
        if y_tl > h_bg or y_tl + h_obj < 0:
            raise Exception(f"The object exceeds the boundaries of the background. y_tl <= h_bg ({y_tl} <= {h_bg}), y_tl + h_obj >= 0 ({y_tl + h_obj} >= 0)") 

        if x_tl < 0: 
            object_img = object_img[:, -x_tl:, :]
            h_obj, w_obj = object_img.shape[:2]
            x_tl = 0
        elif x_tl + w_obj > w_bg:
            diff = x_tl + w_obj - w_bg
            object_img = object_img[:, :w_obj - diff, :]

        if y_tl < 0: 
            object_img = object_img[-y_tl:, :, :]
            h_obj, w_obj = object_img.shape[:2]
            y_tl = 0
        elif y_tl + h_obj > h_bg:
            diff = y_tl + h_obj - h_bg
            object_img = object_img[:h_obj - diff, :, :]

        alpha = object_img[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]

        background_img[y_tl:y_tl+h_obj, x_tl:x_tl+w_obj] = (
            (1 - alpha) * background_img[y_tl:y_tl+h_obj, x_tl:x_tl+w_obj] + alpha * object_img[:, :, :]
        ).astype(np.uint8)

        return background_img
    
    def prepare_coords(self, name: str, coords: Tuple) -> list:
        """Preparing object summary and correct exceeding coordinates.

        Args:
            name (str): object name.
            coords (Tuple): object bounding box.

        Returns:
            dict: object name and bounding box.
        """
        object_to_id_dict: dict = {
            "Cuphead": 0,
            "Platform": 1,
            "Enemy Idle": 2,
            "Enemy FS": 3,
            "Enemy Create": 4,
            "Enemy Final_Idle": 5,
            "Enemy FP": 6,
            "Object Chomper": 7,
            "Object Mini Flower": 8,
            "Object Seed": 9,
            "Object Venus Flytrap": 10,
            "Object Vines": 11,
            "Object Acorn": 12,
            "Object Boomerang": 13,
            "Object Pollen": 14,
            "Object Vines Final": 15,
            "Object Vines Final Platform": 16
        }

        x_tl, y_tl, w_obj, h_obj = coords

        x_tl = max(0, min(x_tl, 640))
        y_tl = max(0, min(y_tl, 360))

        if x_tl + w_obj > 640:
            w_obj_tmp = 640 - x_tl
            x_center: int = x_tl + w_obj_tmp//2
        else:
            x_center: int = x_tl + w_obj//2

        if y_tl + h_obj > 360:
            h_obj_tmp = 360 - y_tl
            y_center: int = y_tl + h_obj_tmp//2
        else:
            y_center: int = y_tl + h_obj//2
                 
        object = [object_to_id_dict[name], x_center, y_center, w_obj, h_obj]

        return object

    def place_platforms(self, background_img: cv2.typing.MatLike) -> Tuple[cv2.typing.MatLike, list]:
        """Generating platforms to the sample.

        Args:
            background_img (cv2.typing.MatLike): background image.

        Returns:
            Tuple[cv2.typing.MatLike, list]: background image with platforms, and platforms coordinates.
        """
        objects_coords: list = list()
        for (x_tl, y_tl) in [(40, 170), (170, 160), (305, 175)]:
            random_sprint_dir: str = random.choice(self.platform_sprints)
            object_img = self.read_image(random_sprint_dir, "platform")
            h_obj, w_obj = object_img.shape[:2]

            y_tl += random.randint(-5, 5)

            background_img = self.paste_image(background_img, object_img, x_tl, y_tl)

            object = self.prepare_coords("Platform", (x_tl, y_tl, w_obj, h_obj))
            objects_coords.append(object)

        return background_img, objects_coords
    

    def place_enemy(self, background_img: cv2.typing.MatLike) -> Tuple[cv2.typing.MatLike, list, str]:
        """Generating enemy to the sample.

        Args:
            background_img (cv2.typing.MatLike): background image.

        Returns:
            Tuple[cv2.typing.MatLike, list, str]: background image with the enemy, the enemy coordinates, the enemy phase name.
        """
        objects_coords: list = list()
        random_sprint_dir: str = random.choice(self.enemy_sprints)
        enemy_phase: str = os.path.basename(random_sprint_dir)
        object_img = self.read_image(random_sprint_dir, "enemy")
        h_obj, w_obj = object_img.shape[:2]

        x_tl: int = 650 - w_obj
        y_tl: int = 350 - h_obj

        background_img = self.paste_image(background_img, object_img, x_tl, y_tl)
        enemy_phase = enemy_phase[:-7]

        if enemy_phase not in ["FS", "Create", "Final_Idle", "FP"]:
            enemy_phase_simplified = "Idle"
        else:
            enemy_phase_simplified = enemy_phase

        object = self.prepare_coords(f"Enemy {enemy_phase_simplified}", (x_tl, y_tl, w_obj, h_obj))
        objects_coords.append(object)

        return background_img, objects_coords, enemy_phase


    def place_object(self, background_img: cv2.typing.MatLike, object_name: str, x_tl_range: Tuple, y_tl_range: Tuple, objects_number: int = 1, use_obj_height: bool = False, use_obj_width: bool = False) -> Tuple[cv2.typing.MatLike, list]:
        """Generating enemy attacks and missiles to the sample.

        Args:
            background_img (cv2.typing.MatLike): background image.
            object_name (str): missiles name
            x_tl_range (Tuple): range of possible occurrence x_tl coordinate.
            y_tl_range (Tuple): range of possible occurrence y_tl coordinate.
            objects_number (int, optional): number of objects to generate. Defaults to 1.
            use_obj_height (bool, optional): flag to use bottom y coordinate instead of top. Defaults to False.
            use_obj_width (bool, optional): flag to use right x coordinate instead of left. Defaults to False.

        Returns:
            Tuple[cv2.typing.MatLike, list]: background image with objects, and objects coordinates.
        """
        objects_coords: list = list()
        for _ in range(objects_number):
            random_sprint_dir: str = random.choice(self.objects_sprints[object_name])
            object_img = self.read_image(random_sprint_dir, "object")
            h_obj, w_obj = object_img.shape[:2]

            x_tl: int = random.randint(*x_tl_range)
            y_tl: int = random.randint(*y_tl_range)

            if use_obj_height:
                y_tl -= h_obj
            if use_obj_width:
                x_tl -= w_obj

            background_img = self.paste_image(background_img, object_img, x_tl, y_tl)

            object = self.prepare_coords(object_name, (x_tl, y_tl, w_obj, h_obj))
            objects_coords.append(object)

        return background_img, objects_coords


    def place_cuphead(self, background_img: cv2.typing.MatLike) -> Tuple[cv2.typing.MatLike, list]:
        """Generating hero to the sample.

        Args:
            background_img (cv2.typing.MatLike): background image.

        Returns:
            Tuple[cv2.typing.MatLike, list]: background image with the hero, and the hero coordinates.
        """
        objects_coords: list = list()
        random_sprint_dir: str = random.choice(self.cuphead_sprints)
        flip: bool = bool(random.getrandbits(1))
        object_img = self.read_image(random_sprint_dir, "cuphead", flip)
        h_obj, w_obj = object_img.shape[:2]

        x_tl: int = random.randint(0, 430)
        y_tl: int = random.randint(0, 240)

        background_img = self.paste_image(background_img, object_img, x_tl, y_tl)

        object = self.prepare_coords("Cuphead", (x_tl, y_tl, w_obj, h_obj))
        objects_coords.append(object)

        return background_img, objects_coords


    def place_objects_logic(self, background_img: cv2.typing.MatLike, enemy_phase: str) -> Tuple[cv2.typing.MatLike, list]:
        """The logic of placing objects at the right moments in combat.

        Args:
            background_img (cv2.typing.MatLike): background image.
            enemy_phase (str): name of the enemy phase.

        Returns:
            Tuple[cv2.typing.MatLike, list]: background image with all objects, and all objects coordinates.
        """
        objects_coords: list = list()
        enemy_phase_dict: dict = {
            "FS": ["Object Chomper", "Object Mini Flower", "Object Seed", "Object Venus Flytrap", "Object Vines"],
            "Create": ["Object Acorn", "Object Boomerang"],
            "Final_Idle": ["Object Vines Final Platform"], #"Object Vines Final"
            "FP": ["Object Pollen", "Object Vines Final Platform"] #"Object Vines Final"
        }
        enemy_attack_dict: dict = {
            "Object Chomper": lambda img: self.place_object(img, "Object Chomper", (0, 480), (260, 260)),
            "Object Mini Flower": lambda img: self.place_object(img, "Object Mini Flower", (100, 280), (25, 25)),
            "Object Seed": lambda img: self.place_object(img, "Object Seed", (40, 380), (0, 250), objects_number=2),
            "Object Venus Flytrap": lambda img: self.place_object(img, "Object Venus Flytrap", (40, 380), (120, 200)),
            "Object Vines": lambda img: self.place_object(img, "Object Vines", (40, 380), (310, 310), use_obj_height=True),
            "Object Acorn": lambda img: self.place_object(img, "Object Acorn", (0, 430), (100, 220), objects_number=3),
            "Object Boomerang": lambda img: self.place_object(img, "Object Boomerang", (0, 640), (130, 190)),
            "Object Pollen": lambda img: self.place_object(img, "Object Pollen", (0, 430), (120, 160)),
            "Object Vines Final": lambda img: self.place_object(img, "Object Vines Final", (430, 430), (300, 300), use_obj_height=True, use_obj_width=True),
            "Object Vines Final Platform": lambda img: self.place_object(img, "Object Vines Final Platform", (40, 305), (300, 300), use_obj_height=True)
        }

        attacks_list: Optional[list] = enemy_phase_dict.get(enemy_phase, None)

        if attacks_list is not None:
            if enemy_phase in ["Final_Idle", "FP"]:
                background_img, object_coords = enemy_attack_dict["Object Vines Final"](background_img)
                objects_coords += object_coords
            
            attack:str = random.choice(attacks_list)
            background_img, object_coords = enemy_attack_dict[attack](background_img)
            objects_coords += object_coords
            
        return background_img, objects_coords
    

    def generate_sample(self, background_img: cv2.typing.MatLike) -> Tuple[cv2.typing.MatLike, list]:
        """Generating sample with all needed occurrences.

        Args:
            background_img (cv2.typing.MatLike): background image.

        Returns:
            Tuple[cv2.typing.MatLike, list]: ready to use sample and objects coordinates.
        """
        objects_coords_list: list = list()
        background_img, platforms_coords = self.place_platforms(background_img)
        background_img, enemy_coords, enemy_phase = self.place_enemy(background_img)
        background_img, objects_coords = self.place_objects_logic(background_img, enemy_phase)
        background_img, cuphead_coords = self.place_cuphead(background_img)

        objects_coords_list = cuphead_coords + enemy_coords + platforms_coords + objects_coords

        return background_img, objects_coords_list


    def create_train_set(self, amount: int) -> None:
        """Generating samples and save them.

        Args:
            amount (int): amount of samples to create.
        """
        background_img = self.read_image(self.battleground_dir, "battleground")
        output_coords: list = list()
        for i in range(amount):
            background_sample_img = background_img.copy()
            background_sample_img, objects_coords_list = self.generate_sample(background_sample_img)
            sample = {
                "image": f"image_{i}.jpg",
                "objects": objects_coords_list
            }

            cv2.imwrite(os.path.join(self.training_set_dir, f"image_{i}.jpg"), background_sample_img)
            output_coords.append(sample)

        with open(os.path.join(self.training_set_dir, "Targets.pickle"), "wb") as output_file:
            pickle.dump(output_coords, output_file)


if __name__ == "__main__":
    sg = SampleGenerator()
    sg.create_train_set(12000)
