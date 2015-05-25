#!/usr/bin/env python
import requests
from json import load
from IPython import embed
from lxml import etree
from io import StringIO, BytesIO
from hashlib import sha1
import logging
import os
import cv2

sources_file = 'sources.json'
cascPath = 'haarcascade_frontalface_default.xml'
logging.basicConfig(level=logging.DEBUG)


sources = load(open(sources_file, 'r'))

# Load html sites
for name, url in sources['www'].iteritems():
    logging.info('Name: %s' % name)
    logging.debug('Url: %s' % url)

    # Hash the site name to get something unique and without space in the title
    m = sha1()
    m.update(name)
    folder = m.hexdigest()

    logging.debug('Folder to store the pictures: %s' % folder)

    # Create the folder if it does not exist
    try: 
        os.makedirs(folder)
    except OSError:
        if not os.path.isdir(folder):
            raise

    # Retreive the site to search for img
    r = requests.get(url)
    if r.status_code != 200:
        logging.error('Error opening a connection to %s' % url)
        logging.error("HTTP %d received" % r.status_code)
        continue        

    # Initialize the HTML parser
    parser = etree.HTMLParser()
    html = etree.parse(StringIO(r.text), parser)
    # search for img in jpg, JPG, jpeg or JPEG
    imgs = [ i for i in html.xpath('//img') if i.attrib['src'][-3:].lower() == 'jpg' or i.attrib['src'][-4:].lower() == 'jpeg' ]

    # For each picture we found on the website
    for img in imgs:
        url = img.attrib['src']
        filename = url.split('/').pop()
        path = '%s/%s' % (folder, filename)

        # Check if the picture already exists
        if os.path.isfile(path): 
            logging.info(" %s already exists" % path)
            continue
        else:
            logging.debug("Downloading %s to %s" % (url, path))
            r = requests.get(url, stream=True)
            chunk_size = 1024
            with open(path, 'wb') as fd:
                for chunk in r.iter_content(chunk_size):
                    fd.write(chunk)
            # Now let's check if there is a face on the picture

            # Create the haar cascade
            faceCascade = cv2.CascadeClassifier(cascPath)
            # Read the image
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(80, 80),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            # If no faces found, remove the picture
            if len(faces) == 0:
                try:
                    logging.info("Removing %s" % path)
                    os.remove(path)
                except:
                    logging.error("Impossible to remove %d" % path)
                continue

            print("Found %d faces" % len(faces))

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            os.remove(path)
            cv2.imwrite(path ,image)
            #cv2.imshow("Faces found" ,image)
            #cv2.waitKey(0)

