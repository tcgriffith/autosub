import sys
import argparse
import pysrt
import wave
import tempfile
import os
import subprocess
import math
import audioop
import datetime
from autosub.constants import LANGUAGE_CODES, \
    GOOGLE_SPEECH_API_KEY, GOOGLE_SPEECH_API_URL
from autosub.formatters import FORMATTERS

def percentile(arr, percent):
    arr = sorted(arr)
    k = (len(arr) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return arr[int(k)]
    d0 = arr[int(f)] * (c - k)
    d1 = arr[int(c)] * (k - f)
    return d0 + d1



def find_speech_regions(filename, frame_width=2000, min_region_size=0.5, max_region_size=6):
    reader = wave.open(filename)
    sample_width = reader.getsampwidth()
    rate = reader.getframerate()
    n_channels = reader.getnchannels()

    total_duration = reader.getnframes() / rate
    chunk_duration = float(frame_width) / rate

    n_chunks = int(math.ceil(reader.getnframes()*1.0 / frame_width))
    energies = []

    for i in range(n_chunks):
        chunk = reader.readframes(frame_width)
        energies.append(audioop.rms(chunk, sample_width * n_channels))

    #threshold = percentile(energies, 0.2)

    elapsed_time = 0

    regions = []
    region_start = None

    for energy in energies:
 #       subenergies=energies[max(0,i-50):min(len(energies)-1,i+50)]
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


def read_region_from_srt(srt_filename):
    # todo
    subs = pysrt.open(srt_filename)
    regions = []
    for asub in subs:
        aregion = (asub.start.ordinal/1000.0,asub.end.ordinal/1000.0)
        regions.append(aregion)
    return regions


def extract_audio(filename, channels=1, rate=16000):
    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    if not os.path.isfile(filename):
        print "The given file does not exist: {0}".format(filename)
        raise Exception("Invalid filepath: {0}".format(filename))

    command = ["ffmpeg", "-y", "-i", filename, "-ac", str(channels), "-ar", str(rate), "-loglevel", "error", temp.name]
    subprocess.check_output(command, stdin=open(os.devnull))
    return temp.name, rate



class SpeechRecognizer(object):
    def __init__(self, language="en", rate=44100, retries=3, api_key=GOOGLE_SPEECH_API_KEY):
        self.language = language
        self.rate = rate
        self.api_key = api_key
        self.retries = retries

    def __call__(self, data):
        try:
            for i in range(self.retries):
                url = GOOGLE_SPEECH_API_URL.format(lang=self.language, key=self.api_key)
                headers = {"Content-Type": "audio/x-flac; rate=%d" % self.rate}

                try:
                    resp = requests.post(url, data=data, headers=headers)
                except requests.exceptions.ConnectionError:
                    continue

                for line in resp.content.split("\n"):
                    try:
                        line = json.loads(line)
                        line = line['result'][0]['alternative'][0]['transcript']
                        return line[:1].upper() + line[1:]
                    except:
                        # no result
                        continue

        except KeyboardInterrupt:
            return


class FLACConverter(object):
    def __init__(self, source_path, include_before=0.25, include_after=0.25):
        self.source_path = source_path
        self.include_before = include_before
        self.include_after = include_after

    def __call__(self, region):
        try:
            start, end = region
            start = max(0, start - self.include_before)
            end += self.include_after
            temp = tempfile.NamedTemporaryFile(suffix='.flac',delete=False)
            command = ["ffmpeg","-ss", str(start), "-t", str(end - start),
                       "-y", "-i", self.source_path,
                       "-loglevel", "error", temp.name]
            subprocess.check_output(command, stdin=open(os.devnull))
            return temp.read()

        except KeyboardInterrupt:
            return


def argparser():
    parser = argparse.ArgumentParser(description='auto sub video or audios')
    parser.add_argument('-i', '--input_path',
                        help="Path to the video or audio file to subtitle")
    parser.add_argument('-c', '--concurrency',
                        help="Number of concurrent Google API requests",
                        type=int,
                        default=10)
    parser.add_argument('-o',
                        '--output',
                        help="output for path of subtitle")
    parser.add_argument('-f', '--format',
                        help="subtitle format",
                        default="srt")
    parser.add_argument('-sl',
                        '--src-language',
                        help="source language",
                        default="en")
    parser.add_argument('-dl', '--dst-language',
                        help="dest language", default="en")
    parser.add_argument('--list-formats',
                        help="List all available subtitle formats",
                        action='store_true')
    parser.add_argument('--list-languages',
                        help="List all available source/destination languages",
                        action='store_true')
    parser.add_argument('-k', '--api-key', help="The google translate API key")

    parser.add_argument('-ts', '--timed_subtitle',
                        help="timed subtitle to replace auto-generated")

    args = parser.parse_args()

    if args.list_formats:
        print("List of formats:")
        for subtitle_format in FORMATTERS.keys():
            print("{format}".format(format=subtitle_format))
        return 0

    if args.list_languages:
        print("List of all languages:")
        for code, language in sorted(LANGUAGE_CODES.items()):
            print("{code}\t{language}".format(code=code, language=language))
        return 0

    if args.format not in FORMATTERS.keys():
        print("Subtitle format not supported. \
              Run with --list-formats to see all supported formats.")
        return 1

    if args.src_language not in LANGUAGE_CODES.keys():
        print("Source language not supported. \
              Run with --list-languages to see all supported languages.")
        return 1

    if args.dst_language not in LANGUAGE_CODES.keys():
        print(
            "Destination language not supported. \
             Run with --list-languages to see all supported languages.")
        return 1

    if not args.input_path:
        print("Error: You need to specify a source path.")
        return 1
    return args

def get_transcripts(regions,args):
    converter = FLACConverter(source_path=audio_filename)
    recognizer = SpeechRecognizer(language=args.src_language, rate=audio_rate, api_key=GOOGLE_SPEECH_API_KEY)

    transcripts = []
    if regions:
        try:
            widgets = ["Converting speech regions to FLAC files: ", Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
            extracted_regions = []
            for i, extracted_region in enumerate(pool.imap(converter, regions)):
                extracted_regions.append(extracted_region)
                pbar.update(i)
            pbar.finish()

            widgets = ["Performing speech recognition: ", Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()

            for i, transcript in enumerate(pool.imap(recognizer, extracted_regions)):
                transcripts.append(transcript)
                pbar.update(i)
            pbar.finish()

        except KeyboardInterrupt:
            pbar.finish()
            pool.terminate()
            pool.join()
            print "Cancelling transcription"
            return 1
    return transcripts



def test():
    srt_file="/home/tc/DATA/subtest/test2.srt"
    regions=read_region_from_srt(srt_file)
    audio_filename, audio_rate = extract_audio(args.source_path)


    transcripts= gettranscripts(regions, audio_filename)



def test2():
    srt_file="/home/tc/DATA/subtest/test2.srt"
    video_file="/home/tc/DATA/subtest/test2.mp4"
    regions=read_region_from_srt(srt_file)
    audio_filename="/tmp/tmp2mMX5L.wav"
#    audio_filename, audio_rate = extract_audio(video_file)


    # start,end=regions[0]

    # temp = tempfile.NamedTemporaryFile(suffix='.flac',delete=False)
    # command = ["ffmpeg","-ss", str(start), "-t", str(end - start),
    #            "-y", "-i", audio_filename,
    #            "-loglevel", "error", temp.name]
    # subprocess.check_output(command, stdin=open(os.devnull))

#    print regions[1][0]

    FLACConverter(audio_filename,regions[0])

def main():
    test2()


if __name__ == '__main__':
    sys.exit(main())
