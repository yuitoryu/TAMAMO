# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 03:14:59 2024

@author: seer2
"""
import sys
import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='This is a demo for preparing dataset from scratch.')
    parser.add_argument('songs', help='Directory of all of your songs (in structure recognizable by AstroDX).')
    parser.add_argument('rating', help='A CSV file that holds songs and their ratings. You can get one from DivingFish')
    parser.add_argument('save_dir', help='Save directory for tokenized chart dataset.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Get the current working directory
    current_working_directory = os.getcwd()
    sys.path.append(current_working_directory+'/tools/')
    from ChartRatingBinder import extract, bind_rate
    from ChartHandler import chartDecomposer, bpmTotimeConverter, noteTokenizer
    
    folder_path = args.songs
    rating_file = args.rating

    rating = bind_rate(rating_file)
    dataset = extract(folder_path, rating)
    processed_data = []
    print('Start extracting and binding rating with charts and primarily decompose the charts.')
    for song in dataset:
        for diff in song['difficulty']:
            fullchart = song['difficulty'][diff]
            current = chartDecomposer()
            current.decompose(fullchart, song['name'])
            this_song_info = current.output_data()
            processed_data.append(this_song_info)
    print('Extracting, binding and decomposing are done.')        
    
    print('Start future handling the charts.')         
    for i in range(len(processed_data)):
        converter = bpmTotimeConverter(processed_data[i])
        tokenizer = noteTokenizer(converter.output())
        #converter = bpmTotimeConverter(processed_data[i])
        processed_data[i] = converter.output()
    

if __name__ == '__main__':
    main()
