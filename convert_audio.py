import librosa
import soundfile as sf
import subprocess
import os
import fleep



def get_audio_type(audio_path):
    with open (audio_path, 'rb') as audio:
        info = fleep.get(audio.read(128))
        print(info.type)
        print(info.extension)
        print(info.mime)

def get_audio_info(audio_path):
    data, samplerate = sf.read(audio_path)
    print('readed')
    print(sf.info(audio_path))
    y, sr = librosa.load(audio_path,sr=16000)
    sf.write('1.wav', y, sr)
    with sf.SoundFile(audio_path, 'r') as audio:
        print("Audio has {} channels.".format(audio.channels))
        return audio.channels

def convert_audio(input_audio, output_audio='output.wav', output_sample_rate=16, bit_rate=256, nb_channels=1):
    """ Convert a given audio to a specific format and type, including MONO/STEREO, bit rate & sample rate.

    Args:
        input_audio (str): Path of the audio to convert.
        output_audio (str): Path and name of the converted audio. Default is 'output.wav'.
        output_sample_rate (int): Sample rate of converted audio in KHz. Default is 16(KHz).
        bit_rate (int): Bit rate of converted audio in Kbps. Default is 256(Kbps).
        nb_channels (int): Number of channels of converted audio (1 for Mono, 2 for Stereo).
                           Default is to 1 (Mono).
    """
    # ffmpeg -i <INPUT_AUDIO> -ac 1 -ab 256k -ar 16k <OUTPUT_AUDIO.wav>

    process = subprocess.run(['ffmpeg', '-i', input_audio, '-ac', str(nb_channels),'-ab', '{}k'.format(bit_rate), '-ar', '{}k'.format(output_sample_rate), output_audio])
    if process.returncode != 0:
        # if something went wrong, try other thing, but dont raise an exception,
        # if eception is raised, you won't know the cause of the fail
        print("Something went wrong")
        # raise Exception("Something went wrong")


def main():
    '''
    1. Read audio
    2. Convert to WAV + MONO + 16KHz
    3. Return audio
    '''

    path_audios = "test_audios"
    for audio in os.listdir(path_audios):
        get_audio_type(os.path.join(path_audios, audio))
        # get_audio_info(os.path.join(path_audios, audio))
        convert_audio(os.path.join(path_audios, audio), audio.rsplit(".", 1)[0]+'.wav')




if __name__ == "__main__":
    main()
