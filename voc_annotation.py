import xml.etree.ElementTree as ET
import os
import glob

sets=[('truck_train'), ('truck_val')]
classes = ['truck']

# sets=[('automatic_train'), ('automatic_val'),
#       ('gun_free_train'), ('gun_free_val'),
#       ('gun_hand_train'), ('gun_hand_val'),
#       ('gun_holstered_train'), ('gun_holstered_val'),
#       ('rifle_carried_train'), ('rifle_carried_val'),
#       ('rifle_shoulder_train'), ('rifle_shoulder_val')]
# classes = ['gun_hand', 'gun_holstered', 'rifle_shoulder', 'rifle_carried', 'automatic', 'gun_free']


def convert_annotation(image_id, list_file):
    in_file = open('data/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), 
            int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = os.getcwd()

image_ids = [os.path.basename(x) for x in glob.glob(os.path.join('data/JPEGImages/*.*'))]
print(image_ids)

list_file = open('%s.txt'%('train_list'), 'w')
for image_id in image_ids:
    list_file.write('data/JPEGImages/{}'.format(image_id))
    convert_annotation('.'.join(image_id.split('.')[0:-1]), list_file)
    list_file.write('\n')
list_file.close()

