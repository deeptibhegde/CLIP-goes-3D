import os

import h5py
import numpy as np
from torch.utils.data import Dataset

__all__ = ['S3DIS']

import random

import torch


def generate_descriptive_caption(class_names):

    captions_dict = {
    'door': [
        'The grand door makes a stunning entrance.',
        'The sleek door blends with the decor.',
        'The door adds sophistication to the room.',
        'The wooden door adds warmth and character.',
        'The frosted glass door provides privacy.',
        'The colorful door adds a playful touch.',
        'The sturdy door provides security.',
        'The double doors create a welcoming entryway.',
        'The sliding door saves space.',
        'The door makes the room appear larger.',
        'The hidden door adds an element mystery.',
        'The barn door adds a rustic charm.',
        'The arched door adds a touch of elegance.',
        'The French doors add a classic feel.',
        'The revolving door adds a modern touch.',
        'The screen door allows for fresh air.',
        'The pocket door maximizes space while maintaining privacy.',
        'The glass door adds a contemporary feel.',
        'The soundproof door provides a quiet environment.',
        'The security door ensures safety.',
    ],
    'chair': [
        'Comfortable chair for lounging and relaxing.',
        'Ergonomic chair for long hours of work.',
        'Elegant dining chair.',
        'Colorful accent chair.',
        'Soothing rocking chair.',
        'Mobile swivel chair.',
        'Practical folding chair.',
        'Luxurious lounge chair.',
        'Trendy bar stool.',
        'Safe high chair for infants and toddlers.',
        'Playful chair addition to any room.',
        'Essential office chair.',
        'Patio chair for outdoor relaxation.',
        'Vintage chair for character and charm.',
        'Supportive armchair for reading or watching TV.',
        "Stylish director's chair for home theater.",
        'Foldable camping chair for outdoor adventures.',
        'Cozy papasan chair.',
        'Timeless wooden rocking chair.',
        'Playful hanging chair seating option.'
    ],
    'table': [
        'Dining table for family and friends.',
        'Coffee table for living room.',
        'End table for decor.',
        'Elegant console table.',
        'Kitchen table for meals.',
        'Versatile folding table.',
        'Outdoor picnic table.',
        'Spacious drafting table.',
        'Classic pool table.',
        'Essential study table.',
        'Outdoor patio table.',
        'Stylish dressing table.',
        'Professional conference table.',
        'Practical bedside table.',
        'Folding picnic table for camping and outdoor adventures.',
        'Rustic farmhouse table for warmth and charm.',
        'Sleek and modern glass table.',
        'Classic card table for game night.',
        'Antique table for character and history.',
        'Adjustable height table for customizable and flexible workspace.'
    ],
    'bookcase': [
         "Bookcase: stylish and practical for organizing books and decor",    
        "Built-in bookcase: custom and seamless touch to decor",    
        "Corner bookcase: maximizes space, provides ample storage",    
        "Ladder bookcase: rustic and industrial touch to decor",    
        "Modular bookcase: versatile and customizable for any room",    
        "Floating bookcase: sleek and minimalist touch to decor",    
        "Wall-mounted bookcase: saves floor space, displays books and decor",   
        "Rotating bookcase: unique and playful element to decor",    
        "Library bookcase: grand and impressive touch to home library",   
        "Metal bookcase: edgy and modern touch to decor",    
        "Narrow bookcase: space-saving solution for small rooms",   
        "Open bookcase: easy access and display of books and decor",  
        "Painted bookcase: colorful and personalized touch to decor", 
        "Wood and metal bookcase: industrial and rustic feel to decor",
        "Glass door bookcase: touch of elegance and sophistication to decor",  
        "Cube bookcase: modern and organized solution for displaying books and decor", 
        "Barrister bookcase: classic and timeless touch to decor",  
        "Tree bookcase: whimsical and playful touch to decor",  
        "Ladder shelf bookcase: space-saving and stylish addition to any room", 
        "Ladder bookcase with desk: functional and efficient workspace solution"],

    "window": [
       "The sunlight streams through the open window.",
        "The rain splatters against the closed window.",
        "The view outside the window is breathtaking.",
        "The dusty window blinds need cleaning.",
        "The window frame is painted a cheerful yellow.",
        "The frost on the window sparkles in the morning light.",
        "The open window lets in a refreshing breeze.",
        "The sheer curtains billow from the window.",
        "The window sill is cluttered with plants and knick-knacks.",
        "The double-paned windows keep out street noise.",
        "The window seat is perfect for reading.",
        "The stained glass window casts colorful light patterns.",
        "The closed window shuts out howling wind.",
        "The bay window is a cozy nook for breakfast.",
        "The window shutters rattle in the wind.",
        "The frosted window film provides privacy and light.",
        "The window overlooks a beautiful garden.",
        "The window in the attic has a stunning city skyline view.",
        "The window screen keeps out bugs on summer nights.",
        "The classroom window provides natural light for studying."
    ],

    "floor": [
        "The polished hardwood floor gleams in the sunlight.",
        "The tiled floor is cool underfoot on a hot summer day.",
        "The plush carpeted floor is a joy to walk on.",
        "The concrete floor of the garage is stained with oil and dirt.",
        "The marble floor of the foyer is impressive and elegant.",
        "The linoleum floor of the kitchen is easy to clean.",
        "The hardwood floor creaks underfoot as you walk through the old house.",
        "The glossy white floor tiles reflect the bright overhead lights.",
        "The parquet floor of the dance studio is perfect for practicing ballet.",
        "The checkered floor of the diner adds to its retro charm.",
        "The bamboo floor of the yoga studio is warm and inviting.",
        "The cobblestone floor of the wine cellar is uneven and rough.",
        "The wooden floorboards of the attic creak as you climb the stairs.",
        "The slate floor of the bathroom is slippery when wet.",
        "The polished concrete floor of the modern loft is sleek and minimalist.",
        "The colorful mosaic floor of the swimming pool is a work of art.",
        "The black and white tile floor of the diner is a classic design.",
        "The natural stone floor of the conservatory is cool and tranquil.",
        "The linoleum floor of the classroom is worn and scuffed.",
        "The hexagonal tile floor of the bathroom is trendy and modern."
    ],

    "sofa": [
        "A comfortable sofa for lounging and relaxing.",
        "The sofa is perfect for a cozy movie night.",
        "This sofa is the centerpiece of the living room.",
        "A modern sofa with sleek lines and soft cushions.",
        "A plush sofa with plenty of room for guests.",
        "A leather sofa with a classic design.",
        "The sofa provides a great spot for conversation.",
        "A cozy sofa that invites you to curl up with a book.",
        "This sectional sofa is perfect for hosting a party.",
        "A stylish sofa that complements the room decor.",
        "The sofa is a versatile piece of furniture.",
        "A comfortable sofa that can double as a bed.",
        "This sofa adds a pop of color to the room.",
        "A spacious sofa with a chaise for ultimate relaxation.",
        "The sofa is the perfect place to take a nap.",
        "A sleek and minimalist sofa that fits any space.",
        "This velvet sofa is luxurious and stylish.",
        "The sofa provides ample seating for a large family.",
        "A cozy and inviting sofa that feels like home.",
        "This sofa has a timeless design that never goes out of style."
    ],
    "board": [
        "A chalkboard for writing notes and reminders.",
        "The whiteboard is perfect for brainstorming ideas.",
        "This cork board helps keep everything organized.",
        "A magnetic board that makes it easy to display photos.",
        "The dry erase board is great for teaching and learning.",
        "A pegboard that can be customized for any need.",
        "This bulletin board is great for displaying art and posters.",
        "A display board for showcasing achievements and awards.",
        "The message board is perfect for leaving notes for loved ones.",
        "A cutting board for meal prep and cooking.",
        "This drawing board is perfect for artists and designers.",
        "A game board for family game night.",
        "The scoreboard keeps track of the game's progress.",
        "A circuit board for electronics projects.",
        "This soundboard is used for mixing and editing audio.",
        "A dart board for a fun and competitive game.",
        "The pedal board is essential for guitarists and musicians.",
        "A surfboard for riding waves and enjoying the ocean.",
        "This board game is fun for players of all ages.",
        "A keyboard for typing and inputting information."
    ],
    "clutter": [
        "The cluttered desk is covered in papers and books.",
        "The kitchen counter is cluttered with dishes and utensils.",
        "The coffee table is covered in clutter, leaving no room for drinks.",
        "The cluttered shelves are filled with books, photos, and trinkets.",
        "The dresser is cluttered with clothes and accessories.",
        "The workbench is cluttered with tools and scraps of wood.",
        "The cluttered bookcase is in desperate need of organization.",
        "The clutter on the floor makes it difficult to walk through the room.",
        "The cluttered closet is bursting with clothes and shoes.",
        "The bathroom sink is cluttered with cosmetics and toiletries.",
        "The dining table is cluttered with dishes and napkins.",
        "The clutter on the nightstand includes a lamp, books, and a phone.",
        "The cluttered living room is in need of a good cleaning.",
        "The cluttered desk drawers are overflowing with office supplies.",
        "The cluttered hallway is filled with shoes, bags, and jackets.",
        "The kitchen pantry is cluttered with boxes, cans, and bags.",
        "The cluttered playroom is filled with toys and games.",
        "The work desk is cluttered with papers, pens, and a computer.",
        "The cluttered garage is full of tools, bikes, and boxes.",
        "The bedside table is cluttered with water bottles, tissues, and a clock."
    ]}

    caption = []
    for class_name in class_names:
        sentence = random.choice(captions_dict[class_name])
        caption.append(sentence)

    return ' '.join(caption)


prompt_templates =[
        "There is a {category} in the scene.",
        "There is the {category} in the scene.",
        "a photo of a {category} in the scene.",
        "a photo of the {category} in the scene.",
        "a photo of one {category} in the scene.",
        "itap of a {category}.",
        "itap of my {category}.",
        "itap of the {category}.",
        "a photo of a {category}.",
        "a photo of my {category}.",
        "a photo of the {category}.",
        "a photo of one {category}.",
        "a photo of many {category}.",
        "a good photo of a {category}.",
        "a good photo of the {category}.",
        "a bad photo of a {category}.",
        "a bad photo of the {category}.",
        "a photo of a nice {category}.",
        "a photo of the nice {category}.",
        "a photo of a cool {category}.",
        "a photo of the cool {category}.",
        "a photo of a weird {category}.",
        "a photo of the weird {category}.",
        "a photo of a small {category}.",
        "a photo of the small {category}.",
        "a photo of a large {category}.",
        "a photo of the large {category}.",
        "a photo of a clean {category}.",
        "a photo of the clean {category}.",
        "a photo of a dirty {category}.",
        "a photo of the dirty {category}.",
        "a bright photo of a {category}.",
        "a bright photo of the {category}.",
        "a dark photo of a {category}.",
        "a dark photo of the {category}.",
        "a photo of a hard to see {category}.",
        "a photo of the hard to see {category}.",
        "a low resolution photo of a {category}.",
        "a low resolution photo of the {category}.",
        "a cropped photo of a {category}.",
        "a cropped photo of the {category}.",
        "a close-up photo of a {category}.",
        "a close-up photo of the {category}.",
        "a jpeg corrupted photo of a {category}.",
        "a jpeg corrupted photo of the {category}.",
        "a blurry photo of a {category}.",
        "a blurry photo of the {category}.",
        "a pixelated photo of a {category}.",
        "a pixelated photo of the {category}.",
        "a black and white photo of the {category}.",
        "a black and white photo of a {category}",
        "a plastic {category}.",
        "the plastic {category}.",
        "a toy {category}.",
        "the toy {category}.",
        "a plushie {category}.",
        "the plushie {category}.",
        "a cartoon {category}.",
        "the cartoon {category}.",
        "an embroidered {category}.",
        "the embroidered {category}.",
        "a painting of the {category}.",
        "a painting of a {category}.",
    ]
def generate_caption(class_names):    
    
    caption = []
    for class_name in class_names:
        prompt = random.choice(prompt_templates)
        sentence = prompt.replace("{category}",class_name)
        caption.append(sentence)

    return ' '.join(caption)



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class _S3DISDataset(Dataset):
    def __init__(self, root, num_points, split='train', with_normalized_coords=False, holdout_area=5):
        """
        :param root: directory path to the s3dis dataset
        :param num_points: number of points to process for each scene
        :param split: 'train' or 'test'
        :param with_normalized_coords: whether include the normalized coords in features (default: True)
        :param holdout_area: which area to holdout (default: 5)
        """
        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.num_points = num_points

        self.real_caption = True
        
        self.scene_list = []

        self.classes = {
            'ceiling': 0,
            'floor': 1,
            'wall': 2,
            'beam': 3,
            'column': 4,
            'window': 5,
            'door': 6,
            'chair': 7,
            'table': 8,
            'bookcase': 9,
            'sofa': 10,
            'board': 11,
            'clutter': 12
        }

        self.class_names = [k for k,v in self.classes.items()]

        self.data_path = os.path.join(self.root,"s3dis/trainval_fullarea/")

        for file in os.listdir(self.data_path):
            self.scene_list.append(os.path.join(self.data_path,file))

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, index):
        filename = self.scene_list[index]
        scene = np.load(filename)

        #remove ceiling
        mask = np.uint8(scene[:,-1]) == 0
        scene = scene[~mask]

        # #remove floor
        # mask =  np.uint8(scene[:,-1]) == 1
        # scene = scene[~mask]

        #remove wall
        mask =  np.uint8(scene[:,-1]) == 2
        scene = scene[~mask]

        #remove beam
        mask =  np.uint8(scene[:,-1]) == 3
        scene = scene[~mask]

        #remove column
        mask =  np.uint8(scene[:,-1]) == 4
        scene = scene[~mask]

        #remove window
        mask =  np.uint8(scene[:,-1]) == 5
        scene = scene[~mask]

        #remove clutter
        mask =  np.uint8(scene[:,-1]) == 12
        scene = scene[~mask]

        #remove board
        mask =  np.uint8(scene[:,-1]) == 11
        scene = scene[~mask]

        #remove door
        mask =  np.uint8(scene[:,-1]) == 6
        scene = scene[~mask]
        

        labels = np.uint8(scene[:,-1])

        scene = pc_normalize(scene[:,:3])
        choice = np.random.choice(scene.shape[0],self.num_points)
        scene = scene[choice]

        

        class_instances = np.unique(labels)

        

        # print(class_instances)

        occurring_classes = []

        for class_instance in class_instances:
            if class_instance != 1:
                occurring_classes.append(self.class_names[class_instance])

        # caption = generate_caption(occurring_classes)
        if self.real_caption:
            caption = generate_descriptive_caption(occurring_classes)
        else:
            caption = generate_caption(occurring_classes)
        #taxonomy_ids, model_ids, points, (img,depth_img), caption, class_name,label
        dummy_img = torch.tensor(0.0)
        return "s3dis","s3dis",scene[:,:3],(dummy_img,dummy_img),caption,'s3dis',index





class S3DIS(dict):
    def __init__(self, root, num_points, split=None, with_normalized_coords=True, holdout_area=5):
        super().__init__()
        if split is None:
            split = ['train', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = _S3DISDataset(root=root, num_points=num_points, split=s,
                                    with_normalized_coords=with_normalized_coords, holdout_area=holdout_area)
