from scipy import misc
import tensorflow as tf
from os.path import isfile, join
import json
import rev
import numpy as np
import pickle

def make_questions_vocab(questions, answers, answer_vocab):
    word_regex = re.compile(r"\w+")
    question_frequency = {}

    max_question_length = 0
    for i,question in enumerate(questions):
        ans = answers[i]["multiple_choice_answer"]
        count = 0
        if ans in answer_vocab:
            question_words = re.findall(word_regex, question["question"])
            for qw in question_words:
                if qw in question_frequency:
                    question_frequency[qw] += 1
                else:
                    question_frequency[qw] = 1
                count += 1
        if count > max_question_length:
            max_question_length = count


    qw_freq_threhold = 0
    qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.items()]

    qw_vocab = {}
    for i, qw_freq in enumerate(qw_tuples):
        frequency = -qw_freq[0]
        qw = qw_freq[1]
        if frequency > qw_freq_threhold:
            qw_vocab[qw] = i + 1
        else:
            break

    qw_vocab["UNK"] = len(qw_vocab) + 1

    return qw_vocab, max_question_length

def make_answer_vocab(answers):
    top_n = 1000
    answer_frequency = {} 
    for annotation in answers:
        answer = annotation["multiple_choice_answer"]
        if answer in answer_frequency:
            answer_frequency[answer] += 1
        else:
            answer_frequency[answer] = 1

    answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.items()]
    answer_frequency_tuples.sort()
    answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

    answer_vocab = {}
    for i, ans_freq in enumerate(answer_frequency_tuples):
        ans = ans_freq[1]
        answer_vocab[ans] = i

    answer_vocab["UNK"] = top_n - 1
    return answer_vocab

def prepare_training_data(data_dir = '../data'):
    
    t_q_json_file = join(data_dir, 'annotations/v2_OpenEnded_mscoco_train2014_questions.json')
    t_a_json_file = join(data_dir, 'annotations/v2_mscoco_train2014_annotations.json')

    v_q_json_file = join(data_dir, 'annotations/v2_OpenEnded_mscoco_val2014_questions.json')
    v_a_json_file = join(data_dir, 'annotations/v2_mscoco_val2014_annotations.json')
    qa_data_file = join(data_dir, 'qa_data_file.pkl')
    vocab_file = join(data_dir, 'vocab_file.pkl')

    if isfile(qa_data_file):
        with open(qa_data_file) as f:
            data = pickle.load(f)
            return data

    print("Loading Training questions")
    with open(t_q_json_file) as f:
        t_questions = json.loads(f.read())
    
    print("Loading Training anwers")
    with open(t_a_json_file) as f:
        t_answers = json.loads(f.read())

    print("Loading Val questions")
    with open(v_q_json_file) as f:
        v_questions = json.loads(f.read())
    
    print("Loading Val answers")
    with open(v_a_json_file) as f:
        v_answers = json.loads(f.read())

    
    print("Ans", len(t_answers['annotations']), len(v_answers['annotations']))
    print("Qu", len(t_questions['questions']), len(v_questions['questions']))

    answers = t_answers['annotations'] + v_answers['annotations']
    questions = t_questions['questions'] + v_questions['questions']
    
    answer_vocab = make_answer_vocab(answers)
    question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab)
    print("Max Question Length", max_question_length)
    word_regex = re.compile(r'\w+')
    training_data = []
    for i,question in enumerate( t_questions['questions']):
        ans = t_answers['annotations'][i]['multiple_choice_answer']
        if ans in answer_vocab:
            training_data.append({
                'image_id' : t_answers['annotations'][i]['image_id'],
                'question' : np.zeros(max_question_length),
                'answer' : answer_vocab[ans]
                })
            question_words = re.findall(word_regex, question['question'])

            base = max_question_length - len(question_words)
            for i in range(0, len(question_words)):
                training_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]

    print("Training Data", len(training_data))
    val_data = []
    for i,question in enumerate( v_questions['questions']):
        ans = v_answers['annotations'][i]['multiple_choice_answer']
        if ans in answer_vocab:
            val_data.append({
                'image_id' : v_answers['annotations'][i]['image_id'],
                'question' : np.zeros(max_question_length),
                'answer' : answer_vocab[ans]
                })
            question_words = re.findall(word_regex, question['question'])

            base = max_question_length - len(question_words)
            for i in range(0, len(question_words)):
                val_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]

    print("Validation Data", len(val_data))

    data = {
        'training' : training_data,
        'validation' : val_data,
        'answer_vocab' : answer_vocab,
        'question_vocab' : question_vocab,
        'max_question_length' : max_question_length
    }

    print("Saving qa_data")
    with open(qa_data_file, 'wb') as f:
        pickle.dump(data, f)

    with open(vocab_file, 'wb') as f:
        vocab_data = {
            'answer_vocab' : data['answer_vocab'],
            'question_vocab' : data['question_vocab'],
            'max_question_length' : data['max_question_length']
        }
        pickle.dump(vocab_data, f)

def load_image_array(image_file):
    img = misc.imread(image_file)
    if len(img.shape) == 2:
        img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype="float32")
        img_new[:, :, 0] = img
        img_new[:, :, 1] = img
        img_new[:, :, 2] = img
        img = img_new

    img_resized = misc.imresize(img, (224, 224))
    return (img_resized / 255.0).astype("float32")

def load_questions_answers(data_dir = 'Data'):
    qa_data_file = join(data_dir, 'qa_data_file.pkl')
    
    if isfile(qa_data_file):
        with open(qa_data_file) as f:
            data = pickle.load(f)
            return data
