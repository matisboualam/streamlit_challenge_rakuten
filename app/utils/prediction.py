import numpy as np
from utils.preprocessing import *



def predict_text(text, text_model, catalog):
    text_input = preprocess_text_data(text)
    pred_text = text_model.predict(text_input)
    predicted_class_index_text = np.argmax(pred_text[0])
    return catalog[predicted_class_index_text]


def predict_image(image_path, im_model, catalog):
    image_input = preprocess_image(image_path)
    pred_im = im_model.predict(image_input)
    predicted_class_index_im = np.argmax(pred_im[0])
    return catalog[predicted_class_index_im]


def predict_fusion(im_model, txt_model, im_acc, txt_acc, article, dataset, class_labels):
    text_input = preprocess_text_data(dataset.text.iloc[article])
    image_input = preprocess_image(f'data/images/{dataset.image_path.iloc[article]}')

    pred_im, pred_txt = None, None

    if im_model is not None:
        pred_im = im_model.predict(image_input, verbose=0)

    if txt_model is not None:
        pred_txt = txt_model.predict(text_input, verbose=0)

    predicted_class_index_im = np.argmax(pred_im[0])
    predicted_class_index_txt = np.argmax(pred_txt[0])

    weighted_pred_im = pred_im[0] * im_acc.values.flatten()
    weighted_pred_txt = pred_txt[0] * txt_acc.values.flatten()

    weighted_pred_im /= np.sum(weighted_pred_im)
    weighted_pred_txt /= np.sum(weighted_pred_txt)

    combined_probs = (weighted_pred_im + weighted_pred_txt) / 2

    final_class_index = np.argmax(combined_probs)

    return {
        'text_input': dataset.text.iloc[article],
        'image_input': dataset.image_path.iloc[article],
        'pred_im': pred_im,
        'pred_txt': pred_txt,
        'weighted_pred_im': weighted_pred_im,
        'weighted_pred_txt': weighted_pred_txt,
        'combined_probs': combined_probs,
        'image_prediction': class_labels[predicted_class_index_im],
        'text_prediction': class_labels[predicted_class_index_txt],
        'final_prediction': class_labels[final_class_index],
        'true_label': dataset.prdtypecode.iloc[article]
    }
