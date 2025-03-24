import os
import re
import numpy as np
import pickle as pkl
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Dropout
from keras import optimizers, metrics, backend as K
from .charactertable import CharacterTable
from tamil import vaaninlp

VAL_MAXLEN = 16

hidden_size = 512
sample_mode = 'argmax'
#data_path = 'models/data/'
data_path = os.path.join(os.path.dirname(__file__), 'models', 'data')

# Pre-load all the files
all_files = os.listdir(data_path)
txt_files = [file for file in all_files if file.endswith('.txt')]
txt_files.sort()

error_rate = 0.6
reverse = True
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models') + '/spellcheck_epoch_20.h5'
SOS = '\t' # start of sequence.
EOS = '*' # end of sequence.
# CHARS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
CHARS = list('அஆஇஈஉஊஎஏஐஒஓஔககாகிகீகுகூகெகேகைகொகோகௌஙஙாஙிஙீஙுஙூஙெஙேஙைஙொஙோஙௌசசாசிசீசுசூசெசேசைசொசோசௌஞஞாஞிஞீஞுஞூஞெஞேஞைஞொஞோஞௌடடாடிடீடுடூடெடேடைடொடோடௌணணாணிணீணுணூணெணேணைணொணோணௌததாதிதீதுதூதெதேதைதொதோதௌநநாநிநீநுநூநெநேநைநொநோநௌபபாபிபீபுபூபெபேபைபொபோபௌமமாமிமீமுமூமெமேமைமொமோமௌயயாயியீயுயூயெயேயையொயோயௌரராரிரீருரூரெரேரைரொரோரௌலலாலிலீலுலூலெலேலைலொலோலௌவவாவிவீவுவூவெவேவைவொவோவௌழழாழிழீழுழூழெழேழைழொழோழௌளளாளிளீளுளூளெளேளைளொளோளௌறறாறிறீறுறூறெறேறைறொறோறௌனனானினீனுனூனெனேனைனொனோனௌஷஷாஷிஷீஷுஷூஷெஷேஷைஷொஷோஷௌஸஸாஸிஸீஸுஸூஸெஸேஸைஸொஸோஸௌஹஹாஹிஹீஹுஹூஹெஹேஹைஹொஹோஹௌ')
REMOVE_CHARS = '[#$%"\+@<=>!&,-.?:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c]'

def read_text(data_path, list_of_books):
    text = ''
    for book in list_of_books:
        file_path = os.path.join(data_path, book)
        print(file_path)
        # strings = unidecode.unidecode(open(file_path).read())
        strings = open(file_path, encoding="utf-8").read()
        text += strings + ' '
    return text

def tokenize(text):
    tokens = [re.sub(REMOVE_CHARS, '', token)
              for token in re.split("[-\n ]", text)]
    return tokens

text  = read_text(data_path, txt_files)
vocab = tokenize(text)
vocab = list(filter(None, set(vocab)))

def add_speling_errors(token, error_rate):
    """Simulate some artificial spelling mistakes."""
    assert(0.0 <= error_rate < 1.0)
    if len(token) < 3:
        return token
    rand = np.random.rand()
    # Here are 4 different ways spelling mistakes can occur,
    # each of which has equal chance.
    prob = error_rate / 4.0
    if rand < prob:
        # Replace a character with a random character.
        random_char_index = np.random.randint(len(token))
        token = token[:random_char_index] + np.random.choice(CHARS) \
                + token[random_char_index + 1:]
    elif prob < rand < prob * 2:
        # Delete a character.
        random_char_index = np.random.randint(len(token))
        token = token[:random_char_index] + token[random_char_index + 1:]
    elif prob * 2 < rand < prob * 3:
        # Add a random character.
        random_char_index = np.random.randint(len(token))
        token = token[:random_char_index] + np.random.choice(CHARS) \
                + token[random_char_index:]
    elif prob * 3 < rand < prob * 4:
        # Transpose 2 characters.
        random_char_index = np.random.randint(len(token) - 1)
        token = token[:random_char_index]  + token[random_char_index + 1] \
                + token[random_char_index] + token[random_char_index + 2:]
    else:
        # No spelling errors.
        pass
    return token

def transform(tokens, maxlen, error_rate=0.3, shuffle=True):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    if shuffle:
        print('Shuffling data.')
        np.random.shuffle(tokens)
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []
    for token in tokens:
        encoder = add_speling_errors(token, error_rate=error_rate)
        encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
        encoder_tokens.append(encoder)

        decoder = SOS + token
        decoder += EOS * (maxlen - len(decoder))
        decoder_tokens.append(decoder)

        target = decoder[1:]
        target += EOS * (maxlen - len(target))
        target_tokens.append(target)

        assert(len(encoder) == len(decoder) == len(target))
    return encoder_tokens, decoder_tokens, target_tokens

# To calculate the truncated accuracy
def truncated_acc(y_true, y_pred):
    y_true = y_true[:, :VAL_MAXLEN, :]
    y_pred = y_pred[:, :VAL_MAXLEN, :]

    acc = metrics.categorical_accuracy(y_true, y_pred)
    return K.mean(acc, axis=-1)

# To calculate the truncated loss
def truncated_loss(y_true, y_pred):
    y_true = y_true[:, :VAL_MAXLEN, :]
    y_pred = y_pred[:, :VAL_MAXLEN, :]

    loss = K.categorical_crossentropy(
        target=y_true, output=y_pred, from_logits=False)
    return K.mean(loss, axis=-1)

def restore_model(path_to_full_model, hidden_size):
    """Restore model to construct the encoder and decoder."""
    model = load_model(path_to_full_model, compile=False, custom_objects={
        'truncated_acc': truncated_acc, 'truncated_loss': truncated_loss})
    model.compile()
    
    encoder_inputs = model.input[0] # encoder_data
    encoder_lstm1 = model.get_layer('encoder_lstm_1')
    encoder_lstm2 = model.get_layer('encoder_lstm_2')

    encoder_outputs = encoder_lstm1(encoder_inputs)
    _, state_h, state_c = encoder_lstm2(encoder_outputs)
    encoder_states = [state_h, state_c]
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

    decoder_inputs = model.input[1] # decoder_data
    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_softmax = model.get_layer('decoder_softmax')
    decoder_outputs = decoder_softmax(decoder_outputs)
    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
                          outputs=[decoder_outputs] + decoder_states)
    return encoder_model, decoder_model

def batch(tokens, maxlen, ctable, batch_size=128, reverse=False):
    """Split data into chunks of `batch_size` examples."""
    def generate(tokens, reverse):
        while(True): # This flag yields an infinite generator.
            for token in tokens:
                if reverse:
                    token = token[::-1]
                yield token

    token_iterator = generate(tokens, reverse)
    data_batch = np.zeros((batch_size, maxlen, ctable.size),
                          dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = ctable.encode(token, maxlen)
        yield data_batch

def decode_sequences(inputs, targets, input_ctable, target_ctable,
                     maxlen, reverse, encoder_model, decoder_model,
                     nb_examples, sample_mode='argmax', random=True):
    input_tokens = []
    target_tokens = []

    if random:
        indices = np.random.randint(0, len(inputs), nb_examples)
    else:
        indices = range(nb_examples)

    for index in indices:
        input_tokens.append(inputs[index])
        target_tokens.append(targets[index])
    input_sequences = batch(input_tokens, maxlen, input_ctable,
                            nb_examples, reverse)
    input_sequences = next(input_sequences)

    # Procedure for inference mode (sampling):
    # 1) Encode input and retrieve initial decoder state.
    # 2) Run one step of decoder with this initial state
    #    and a start-of-sequence character as target.
    #    Output will be the next target character.
    # 3) Repeat with the current target character and current states.

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_sequences)

    # Create batch of empty target sequences of length 1 character.
    target_sequences = np.zeros((nb_examples, 1, target_ctable.size))
    # Populate the first element of target sequence
    # with the start-of-sequence character.
    target_sequences[:, 0, target_ctable.char2index[SOS]] = 1.0

    # Sampling loop for a batch of sequences.
    # Exit condition: either hit max character limit
    # or encounter end-of-sequence character.
    decoded_tokens = [''] * nb_examples
    for _ in range(maxlen):
        # `char_probs` has shape
        # (nb_examples, 1, nb_target_chars)
        char_probs, h, c = decoder_model.predict(
            [target_sequences] + states_value)

        # Reset the target sequences.
        target_sequences = np.zeros((nb_examples, 1, target_ctable.size))

        # Sample next character using argmax or multinomial mode.
        sampled_chars = []
        for i in range(nb_examples):
            if sample_mode == 'argmax':
                next_index, next_char = target_ctable.decode(
                    char_probs[i], calc_argmax=True)
            elif sample_mode == 'multinomial':
                next_index, next_char = target_ctable.sample_multinomial(
                    char_probs[i], temperature=0.5)
            else:
                raise Exception(
                    "`sample_mode` accepts `argmax` or `multinomial`.")
            decoded_tokens[i] += next_char
            sampled_chars.append(next_char)
            # Update target sequence with index of next character.
            target_sequences[i, 0, next_index] = 1.0

        stop_char = set(sampled_chars)
        if len(stop_char) == 1 and stop_char.pop() == EOS:
            break

        # Update states.
        states_value = [h, c]

    # Sampling finished.
    input_tokens   = [re.sub('[%s]' % EOS, '', token)
                      for token in input_tokens]
    target_tokens  = [re.sub('[%s]' % EOS, '', token)
                      for token in target_tokens]
    decoded_tokens = [re.sub('[%s]' % EOS, '', token)
                      for token in decoded_tokens]
    return input_tokens, target_tokens, decoded_tokens

def compare_sentences(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()
    compared_sentence = ""

    for original_word in original_words:
        # If the word contains a punctuation mark, handle it separately
        if any(char in ",.!?;:()[]{}<>/\\&@#%$€£¥^*+=|~_-" for char in original_word):
            # Separate punctuation from the word
            word_without_punctuation = ''.join(char for char in original_word if char not in ",.!?;:()[]{}<>/\\&@#%$€£¥^*+=|~_-")
            punctuation = ''.join(char for char in original_word if char in ",.!?;:()[]{}<>/\\&@#%$€£¥^*+=|~_-")
            compared_sentence += f"{word_without_punctuation}{punctuation} "
        else:
            if corrected_words:
                corrected_word = corrected_words.pop(0)
                if original_word != corrected_word:
                    compared_sentence += f"<span class='highlight'>{corrected_word}</span> "
                else:
                    compared_sentence += f"{original_word} "
            else:
                compared_sentence += f"{original_word} "  # If corrected words run out, retain original

    return compared_sentence.strip()

PUNCTUATIONS = ",.!?;:()[]{}<>/\\&@#%$€£¥^*+=|~_-"
def fallback(text):
    sentences = ''
    for line in text.split('.'):
        if len(line) == 0:
            continue
        
        line += '.'
        words = line.strip().split(' ')
        tokens = vaaninlp.word_tokenize(line)
        result = vaaninlp.spellcheck(tokens)

        sentence = ''
        print(result)
        for cword, orig_word in zip(result, words):
            print(cword)
            print(orig_word)
            if not cword['Flag']:
                print(cword['Suggestions'])
                suggested_word = cword['Suggestions'].split(',')[0]
                # Compare the suggested word with the original word - Check if there are any punctuations and add them
                for char in orig_word:
                    if char in PUNCTUATIONS:
                        suggested_word += char
                sentence += '<span class="highlight">' + suggested_word + '</span> '
            else:
                # Compare the suggested word with the original word - Check if there are any punctuations and add them
                sentence += orig_word + ' '
        # sentence = compare_sentences(line, sentence.strip()).strip()
        sentences += sentence
    return sentences

def do_check(text):
    return fallback(text).strip()
    # `maxlen` is the length of the longest word in the vocabulary
    # plus two SOS and EOS characters.
    maxlen = max([len(token) for token in vocab]) + 2
    with open(os.path.join(os.path.dirname(__file__), 'models') + '/train_encoder.pkl', 'rb') as fp:
        train_encoder = pkl.load(fp)
    with open(os.path.join(os.path.dirname(__file__), 'models') + '/train_encoder.pkl', 'rb') as fp:
        train_decoder = pkl.load(fp)
    with open(os.path.join(os.path.dirname(__file__), 'models') + '/train_encoder.pkl', 'rb') as fp:
        train_target = pkl.load(fp)
    
    # train_encoder, train_decoder, train_target = transform(
    #     vocab, maxlen, error_rate=error_rate, shuffle=False)
    sentences = ''
    print('Starting')
    for line in text.split('.'):
        print(line)
        if len(line) == 0:
            continue

        tokens = tokenize(line)
        tokens = list(filter(None, tokens))
        nb_tokens = len(tokens)
        misspelled_tokens, _, target_tokens = transform(
            tokens, maxlen, error_rate=error_rate, shuffle=False)

        with open(os.path.join(os.path.dirname(__file__), 'models') + '/input_chars.pkl', 'rb') as fp:
            input_chars = pkl.load(fp)
        with open(os.path.join(os.path.dirname(__file__), 'models') + '/target_chars.pkl', 'rb') as fp:
            target_chars = pkl.load(fp)
        # input_chars = set(' '.join(train_encoder))
        # target_chars = set(' '.join(train_decoder))
        input_ctable = CharacterTable(input_chars)
        target_ctable = CharacterTable(target_chars)

        encoder_model, decoder_model = restore_model(MODEL_PATH, hidden_size)

        input_tokens, target_tokens, decoded_tokens = decode_sequences(
            misspelled_tokens, target_tokens, input_ctable, target_ctable,
            maxlen, reverse, encoder_model, decoder_model, nb_tokens,
            sample_mode=sample_mode, random=False)

        print('-')
        print('Input sentence:  ', ' '.join([token for token in input_tokens]))
        print('-')
        print('Decoded sentence:', ' '.join([token for token in decoded_tokens]))
        print('-')
        print('Target sentence: ', ' '.join([token for token in target_tokens]))

        sentence = ' '.join([token for token in decoded_tokens])
        sentence = compare_sentences(line, sentence).strip()
        sentences += sentence
    return sentences