"""Safely replace one LBPH identity while preserving all other identities.

Only histograms bearing the chosen face_label are removed. Existing raw face
images aren't needed; the legacy trainer.yml already stores the LBPH features.
Requires OpenCV's FileStorage schema used by LBPHFaceRecognizer. Fails closed
on schema changes and leaves the original model untouched.
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.face_enrollment import MODEL_PATH, need_cv2_contrib, validate_capture


def _read_lbph_file(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not fs.isOpened():
            raise ValueError('Cannot open the stored face model.')
        node = fs.getNode('opencv_lbphfaces')
        if node.empty():
            raise ValueError('Unrecognized LBPH trainer.yml format; no changes made.')
        settings = {}
        for field in ('radius', 'neighbors', 'grid_x', 'grid_y', 'threshold'):
            child = node.getNode(field)
            if child.empty():
                raise ValueError(f'LBPH trainer.yml missing {field}; no changes made.')
            settings[field] = (float(child.real()) if field == 'threshold'
                               else int(child.real()))
        labels_node = node.getNode('labels')
        hist_node = node.getNode('histograms')
        if labels_node.empty() or hist_node.empty():
            raise ValueError('LBPH trainer.yml missing labels/histograms.')
        label_matrix = labels_node.mat()
        if label_matrix is None:
            raise ValueError('LBPH trainer.yml contains no labels.')
        labels = np.asarray(label_matrix).reshape(-1).astype(np.int32)
        histograms = [hist_node.at(i).mat() for i in range(hist_node.size())]
        if not len(labels) or len(labels) != len(histograms):
            raise ValueError('Face template count does not match face labels.')
        if any(x is None or not x.size for x in histograms):
            raise ValueError('One of the stored face templates is invalid.')
        return settings, labels, histograms
    finally:
        fs.release()


def _write_lbph_file(path, settings, labels, histograms):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError('Unable to stage the updated face model.')
    try:
        fs.startWriteStruct('opencv_lbphfaces', cv2.FileNode_MAP)
        for field in ('threshold', 'radius', 'neighbors', 'grid_x', 'grid_y'):
            fs.write(field, settings[field])
        fs.startWriteStruct('histograms', cv2.FileNode_SEQ)
        for histogram in histograms:
            fs.write('', np.asarray(histogram, dtype=np.float32))
        fs.endWriteStruct()
        fs.write('labels', np.asarray(labels, dtype=np.int32).reshape(1, -1))
        fs.endWriteStruct()
    finally:
        fs.release()


def build_replacement_model(samples, face_label, model_path=MODEL_PATH):
    """Return path to checked staged trainer; DOES NOT replace model_path.

    Keep every other face's LBPH histograms, swap ONLY the target identity's
    histograms for validated new scan histograms, and preserve the label number.
    Caller owns the returned file and must unlink it if not published.
    """
    need_cv2_contrib()
    validate_capture(samples)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError('No existing trainer.yml. Cannot safely preserve other faces.')
    face_label = int(face_label)

    settings, labels, histograms = _read_lbph_file(model_path)
    if face_label not in labels:
        raise ValueError('The selected account has no templates in trainer.yml.')

    # An update must not enroll the same face under a different account.
    old_rec = cv2.face.LBPHFaceRecognizer_create()
    old_rec.read(str(model_path))
    for sample in samples[::3]:
        matched_label, matched_distance = old_rec.predict(sample)
        if int(matched_label) != face_label and float(matched_distance) < 55:
            raise ValueError('Face may already belong to another administrator.')

    rec = cv2.face.LBPHFaceRecognizer_create(
        settings['radius'], settings['neighbors'],
        settings['grid_x'], settings['grid_y'], settings['threshold'])
    rec.train(samples, np.full(len(samples), face_label, dtype=np.int32))
    handle, temp_new = tempfile.mkstemp(prefix='.spot-fresh-', suffix='.yml',
                                         dir=str(model_path.parent))
    os.close(handle)
    handle, temp_merged = tempfile.mkstemp(prefix='.spot-replaced-', suffix='.yml',
                                            dir=str(model_path.parent))
    os.close(handle)
    try:
        rec.write(temp_new)
        _new_settings, new_labels, new_hists = _read_lbph_file(temp_new)
        keep_mask = labels != face_label
        kept_labels = list(labels[keep_mask])
        kept_histograms = [hist for hist, keep in zip(histograms, keep_mask) if keep]
        # Guard against accidental re-training/alteration of other identities.
        all_labels = np.asarray(kept_labels + list(new_labels), dtype=np.int32)
        all_hists = kept_histograms + new_hists
        original_others = set(int(x) for x in labels if x != face_label)
        if set(int(x) for x in all_labels if x != face_label) != original_others:
            raise RuntimeError('Cannot preserve all other administrators; model unchanged.')
        _write_lbph_file(temp_merged, settings, all_labels, all_hists)
        check = cv2.face.LBPHFaceRecognizer_create()
        check.read(temp_merged)
        verified_labels = [int(x) for x in check.getLabels().reshape(-1)]
        if sorted(verified_labels) != sorted(int(x) for x in all_labels):
            raise RuntimeError('Saved model labels failed validation.')
        predicted_label, distance = check.predict(samples[-1])
        if int(predicted_label) != face_label or float(distance) >= 100:
            raise RuntimeError('New face was not recognized by the replacement model.')
        return Path(temp_merged)
    except Exception:
        Path(temp_merged).unlink(missing_ok=True)
        raise
    finally:
        Path(temp_new).unlink(missing_ok=True)
