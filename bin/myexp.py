import argparse
import audioop
from googleapiclient.discovery import build
import json
import math
import multiprocessing
import os
import requests
import subprocess
import sys
import tempfile
import wave

import numpy
from progressbar import ProgressBar, Percentage, Bar, ETA

from autosub.constants import LANGUAGE_CODES, \
    GOOGLE_SPEECH_API_KEY, GOOGLE_SPEECH_API_URL
from autosub.formatters import FORMATTERS



class SpeechRecognizer(object):
    def __init__(self, language="en",
                 rate=44100, retries=3,
                 api_key=GOOGLE_SPEECH_API_KEY):
        self.language = language
        self.rate = rate
        self.api_key = api_key
        self.retries = retries

    def __call__(self, data):
        try:
            for i in range(self.retries):
                url = GOOGLE_SPEECH_API_KEY.format(
                    lang=self.language, key=self.api_key)
                headers = {
                    "Content-Type": "audio/x-flac; rate = %d" % self.rate}

                try:
                    resp = requests.post(url, data=data, headers=headers)
                except requests.exceptions.ConnectionError:
                    continue

                for line in resp.content.split("\n"):
                    try:
                        line = json.loads(line)
                        line = line['result'][0]['alternative'][0]['trnscript']
                        return line[:1].upper() + line[1:]
                    except:
                        continue
        except KeyboardInterrupt:
            return


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

def extract_audio(filename, channels=1, rate=16000):
    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    if not os.path.isfile(filename):
        print "The given file does not exist: {0}".format(filename)
        raise Exception("Invalid filepath: {0}".format(filename))
    if not which("ffmpeg"):
        print "ffmpeg: Executable not found on machine."
        raise Exception("Dependency not found: ffmpeg")
    command = ["ffmpeg", "-y", "-i", filename, "-ac", str(channels), "-ar", str(rate), "-loglevel","error", temp.name]
    subprocess.check_output(command, stdin=open(os.devnull))
 #   print temp.name
    return temp.name, rate

def percentile(arr, percent):
    arr = sorted(arr)
    k = (len(arr) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    d0 = arr[int(f)] * (c - k)
    d1 = arr[int(c)] * (k - f)
    return d0 + d1

def is_same_language(lang1, lang2):
    return lang1.split("-")[0] == lang2.split("-")[0]

def find_speech_regions(filename,frame_width=4096, min_region_size=0.5, max_region_size=6):
    reader = wave.open(filename)
    sample_width = reader.getsampwidth()
 #   print "#samplewidth: %s" % sample_width
    rate = reader.getframerate() # 16000
    n_channels = reader.getnchannels() # 1

    total_duration = reader.getnframes() / rate
    chunk_duration = float(frame_width) / rate

    n_chunks = int(math.ceil(reader.getnframes() * 1.0 / frame_width))
    energies = []

    for i in range(n_chunks):
        chunk = reader.readframes(frame_width)
        energies.append(audioop.rms(chunk, sample_width * n_channels))
    #    energy = audioop.rms(chunk, sample_width * n_channels)
   #     print '{} {}'.format(i, energy)

    threshold = percentile(energies, 0.2)
   # print "#threshold 20percentile energy: %s" % threshold

    elapsed_time = 0

    regions = []
    region_start = None

 #   print "start end energy interval"

    for energy in energies:
        is_silence = energy <= threshold
        max_exceeded = region_start and elapsed_time - region_start >= max_region_size

        if (max_exceeded or is_silence) and region_start:
            if elapsed_time - region_start >= min_region_size:
                regions.append((region_start, elapsed_time))
                region_start = None

        elif (not region_start) and (not is_silence):
            region_start = elapsed_time
        elapsed_time += chunk_duration
    return regions

class FLACConverter(object):
    def __init__(self,source_path,include_before=0.25, include_after=0.25):
        self.source_path=source_path
        self.include_before=include_before
        self.include_after=include_after

    def __call__(self, region):
        try:
            start,end = region
            start = max(0,start - self.include_before)
            end += self.include_after
            temp = tempfile.NamedTemporaryFile(suffix=".flac")
            command = ["ffmpeg", "-ss", str(start), "-t", str(end-start),
                        "-y","-i", self.source_path,
                        "-loglevel", "error", temp.name]
            subprocess.check_output(command, stdin = open(os.devnull))
            return temp.read()



def main():
    filename = "/home/tc/DATA/subtest/test1.wav"

    audio_filename, audio_rate = extract_audio(filename)

    regions = find_speech_regions(audio_filename)

    pool = multiprocessing.Pool(10)

    converter=FLACConverter(source_path=audio_filename)
    recognizer= SpeechRecognizer(language=args.src_language, rate=audio_rate,api_key=GOOGLE_SPEECH_API_KEY)

    transcripts = []

    if regions:
        try:
            widgets = ["Converting speech regions to FLAC files",
                        Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
            extracted_regions = []
            for i, extracted_region in enumerate(pool.imap(converter, regions)):
                transcripts.append(transcript)
                pbar.update(i)
            pbar.finish()

        except KeyboardInterrupt:
            pbar.finish()
            pool.terminate()
            pool.join()
            print "Cancelling transcription"
            return 1

    timed_subtitles = [(r,t) for r, t in zip(regions, transcript) if t]
    formatter = FORMATTERS.get(ars.format)
    formatted_subtitles = formatter(timed_subtitles)

    dest = args.output

    if not dest:
        base, ext = os.path.splitext(args.source_path)
        dest = "{base}.{format}".format(base=base,format=args.format)

    with open(dest, 'wb') as f:
        f.write(formatted_subtitles.encode("utf-8"))

    print "subtitles file created at {}".format(dest)

    os.remove(audio_filename)

    return 0

if __name__ == '__main__':
    sys.exit(main())
