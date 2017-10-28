# -*- coding: utf-8 -*-
import sys
import time
import subprocess as cmd
import http.client
import json
import re
import cv2
import requests
import os
import math
import pickle
import urllib
import hashlib
import sha3
import copy
from argparse import ArgumentParser

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def make_correspondence_table(correspondence_table, original_url, hashed_url):
    """Create reference table of hash value and original URL.
    """
    correspondence_table[original_url] = hashed_url


def make_img_path(save_dir_path, url):
    """Hash the image url and create the path

    Args:
        save_dir_path (str): Path to save image dir.
        url (str): An url of image.

    Returns:
        Path of hashed image URL.
    """
    save_img_path = os.path.join(save_dir_path, 'imgs')
    make_dir(save_img_path)

    file_extension = os.path.splitext(url)[-1]
    if file_extension.lower() in ('.jpg', '.jpeg', '.gif', '.png', '.bmp'):
        encoded_url = url.encode('utf-8') # required encoding for hashed
        hashed_url = hashlib.sha3_256(encoded_url).hexdigest()
        full_path = os.path.join(save_img_path, hashed_url + file_extension.lower())

        make_correspondence_table(correspondence_table, url, hashed_url)

        return full_path
    else:
        raise ValueError('Not applicable file extension...{}'.format(file_extension))

def make_img_path2(save_dir_path, i, N, num_imgs_per_transaction):
    save_img_path = os.path.join(save_dir_path, 'imgs')
    original_img_path = os.path.join(save_img_path, 'Original')
    make_dir(save_img_path)
    make_dir(original_img_path)

    #パスと拡張子に分割(例;http://~~/img1.txt=>http://~~/img+.txt)
    file_extenxion = os.path.splitext(url)[-1]
    #lower小文字に変換
    if file_extenxion.lower() in ('.jpg', '.jpeg', '.gif', '.png', '.bmp'):
        full_path = os.path.join(original_img_path, str(i+N*num_imgs_per_transaction) + file_extenxion.lower())
        return full_path
    else:
        raise ValueError('Not applicable file extension')

def download_image(url, timeout=10):
    # print("downloading..."+url)
    response = requests.get(url, allow_redirects=True, timeout=timeout)
    print(str(response.status_code))
    if response.status_code != 200:
        error = Exception("HTTP status: " + response.status_code)
        raise error

    content_type = response.headers["content-type"]
    if 'image' not in content_type:
        error = Exception("Content-Type: " + content_type)
        raise error

    return response.content


def save_image(filename, image):
    with open(filename, "wb") as fout:
        fout.write(image)

def cropFace(save_base_dir, path, imsize, method):
    cropped_img_path = os.path.join(save_base_dir, 'Cropped')
    dir = cropped_img_path
    if not (os.path.exists(dir)):
        os.mkdir(dir)
    original_img_path = os.path.join(save_base_dir, 'Original')
    for p in path:
        img = cv2.imread(original_img_path + '/' + p)
        try:
            gImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except Exception as err:
            # print("[Errno {0}] {1}".format(err.errno, err.strerror))
            continue

        if method == 1:
            face_cascade = cv2.CascadeClassifier('/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        elif method == 2:
            face_cascade = cv2.CascadeClassifier('/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
        elif method == 3:
            face_cascade = cv2.CascadeClassifier('/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
        else:
            face_cascade = cv2.CascadeClassifier('/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml')
        faces = face_cascade.detectMultiScale(gImg, 1.3, 5)
        for num in range(len(faces)):
            cropImg = copy.deepcopy(img[faces[num][1]:faces[num][1]+faces[num][3], faces[num][0]:faces[num][0]+faces[num][2]])
            resizeImg = cv2.resize(cropImg, (imsize, imsize))
            filename = dir + '/' + p[:-4] + '_' + str(num + 1) + '.tif'
            cv2.imwrite(filename, resizeImg)

if __name__ == "__main__":
    
    ap = ArgumentParser(description='bing_search.py')
    ap.add_argument('--query', '-q', nargs='*', default='hoge', help='Specify Query of Image Collection ')
    ap.add_argument('--suffix', '-s', nargs='*', default='jpg', help='Specify Image Suffix')
    ap.add_argument('--imsize', '-i', type=int, default=100, help='Specify Image Size of Crop Face Image')
    ap.add_argument('--method', '-m', type=int, default=1, help='Specify Method Flag (1 : Haarcascades Frontalface Default, 2 : Haarcascades Frontalface Alt1, 3 : Haarcascades Frontalface Alt2, Without : Haarcascades Frontalface Alt Tree)')
    args = ap.parse_args()

    # Set Yout Apikey
    api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    
    t = time.ctime().split(' ')
    if t.count('') == 1:
        t.pop(t.index(''))
    # Path Separator
    psep = '/'
    for q in args.query:
        opbase = q
        # Delite File Sepaeator   
        if (opbase[len(opbase) - 1] == psep):
            opbase = opbase[:len(opbase) - 1]
        # Add Current Directory (Exept for Absolute Path)
        if not (opbase[0] == psep):
            if (opbase.find('./') == -1):
                make_dir('./output')
                opbase = './output/' + opbase
        # Create Opbase
        opbase = opbase + '_' + t[1] + t[2] + t[0] + '_' + t[4] + '_' + t[3].split(':')[0] + t[3].split(':')[1] + t[3].split(':')[2]
        if not (os.path.exists(opbase)):
            # os.mkdir(opbase)
            print ('Output Directory not exist! Create...')
        print ('Output Directory: ' + opbase)

        save_dir_path = opbase
        make_dir(save_dir_path)

        num_imgs_required = 1000 # Number of images you want. The number to be divisible by 'num_imgs_per_transaction'
        num_imgs_per_transaction = 150 # default 30, Max 150
        offset_count = math.floor(num_imgs_required / num_imgs_per_transaction)

        url_list = []
        correspondence_table = {}

        headers = {
            # Request headers
            # 'Content-Type': 'multipart/form-data',
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': api_key, # API key
        }

        for offset in range(offset_count):

            params = urllib.parse.urlencode({
                # Request parameters
                'ubscription-Key': api_key,
                'q': q,
                'mkt': 'ja-JP',
                'count': num_imgs_per_transaction,
                'offset': offset * num_imgs_per_transaction # increment offset by 'num_imgs_per_transaction' (for example 0, 150, 300)
            })

            try:
                conn = http.client.HTTPSConnection('api.cognitive.microsoft.com')
                conn.request("GET", "/bing/v7.0/images/search?%s" % params, "{body}", headers)
                response = conn.getresponse()
                data = response.read()

                save_res_path = os.path.join(save_dir_path, 'pickle_files')
                make_dir(save_res_path)
                with open(os.path.join(save_res_path, '{}.pickle'.format(offset)), mode='wb') as f:
                    pickle.dump(data, f)

                conn.close()
            except Exception as err:
                print("[Errno {0}] {1}".format(err.errno, err.strerror))

            else:
                decode_res = data.decode('utf-8')
                data = json.loads(decode_res)
                # print(data)
                pattern = r"&r=(http.+)&p=" # extract an URL of image

                for values in data['value']:
                    unquoted_url = urllib.parse.unquote(values['contentUrl'])
                    img_url = re.search(pattern, unquoted_url)
                    if img_url:
                        url_list.append(img_url.group(1))
                        print("list added: "+img_url.group(1))
                    else:
                        url_list.append(unquoted_url)
                        # print("list added: "+unquoted_url)

            num_of_list = len(url_list)
            print("count of url: "+str(num_of_list))

            for i, url in enumerate(url_list):
                try:
                    # img_path = make_img_path(save_dir_path, url)
                    img_path = make_img_path2(save_dir_path, i, offset, num_imgs_per_transaction)
                    image = download_image(url)
                    # print('image downloaded...{}'.format(url))
                    save_image(img_path, image)
                    print('saved image... {}'.format(url))
                except KeyboardInterrupt:
                    break
                except Exception as err:
                    print("%s" % (err))

        save_image_dir_path = save_dir_path + '/imgs'
        cropFace(save_image_dir_path, os.listdir(save_image_dir_path + '/Original'), args.imsize, args.method)

        correspondence_table_path = os.path.join(save_dir_path, 'corr_table')
        make_dir(correspondence_table_path)

        with open(os.path.join(correspondence_table_path, 'corr_table.json'), mode='w') as f:
            json.dump(correspondence_table, f)