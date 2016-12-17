import tensorflow as tf
import sys
import os 

# change this as you see fit
folder_path = sys.argv[1]

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

out = []
output = {}
global img_path
with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    #read individual files in folder and tabulate accuracy
    for cat in os.listdir(folder_path):
        output = {cat:99}
        if "." not in cat:
            for img in os.listdir(folder_path+"/"+cat):
                if img.endswith(".jpg"):                
                    img_path = folder_path+"/"+cat+"/"+img
                    image_data = tf.gfile.FastGFile(img_path, 'rb').read()
                        
                    predictions = sess.run(softmax_tensor, \
                             {'DecodeJpeg/contents:0': image_data})
                
                    # Sort to show labels of first prediction in order of confidence
                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                    pred = label_lines[top_k[0]]
                    score = predictions[0][top_k[0]]
                    score = str(score)
                    if pred not in output:
                        output[pred] = 1
                    else:
                        output[pred] += 1

                    rename = 'pred = '+pred + ","+' score = ' + ""+score+'.jpg'

                    os.rename(img_path, folder_path+"/"+cat+"/"+rename)
        print(output)

