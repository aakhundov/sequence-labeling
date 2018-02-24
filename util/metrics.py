import numpy as np


def extract_entities(data_labels, seq_len, num_labels, num_classes):
    with_classes = set([])
    without_classes = set([])

    for i in range(len(data_labels)):
        current = None
        sentence = data_labels[i][:seq_len[i]]
        for l in range(len(sentence)):
            lbl = sentence[l]
            if current is not None and \
               (lbl < num_classes or lbl == num_labels - 1 or current[2] != lbl - num_classes):
                with_classes.add(current + (l-1,))
                without_classes.add(current[:-1] + (l-1,))
                current = None
            if lbl < num_classes:
                current = (i, l, lbl)
            elif lbl < num_labels - 1 and current is None:
                current = (i, l, lbl-num_classes)
        if current is not None:
            with_classes.add(current + (len(sentence)-1,))
            without_classes.add(current[:-1] + (len(sentence)-1,))

    return with_classes, without_classes


def compute_f1_score(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec * 100, rec * 100, f1 * 100


def are_iob_labels(label_names):
    return all(lb[:2] in ["B-", "I-"] or lb == "O" for lb in label_names)


def are_iobes_labels(label_names):
    return all(lb[:2] in ["B-", "I-", "E-", "S-"] or lb == "O" for lb in label_names) and \
           any(lb[:2] in ["E-", "S-"] for lb in label_names)


def f1_scores_required(labels, num_labels, seq_len, min_o_label_ratio=0.8):
    num_o_labels = sum([sum([
        1 for lb in labels[i][:seq_len[i]]
        if lb == num_labels - 1
    ]) for i in range(len(seq_len))])
    return num_o_labels / sum(seq_len) >= min_o_label_ratio


def iobes_to_iob(gold, predicted, label_names, seq_len):
    num_classes = (len(label_names) - 1) // 4
    gold, predicted = gold[:], predicted[:]

    for matrix in [gold, predicted]:
        for i in range(len(matrix)):
            for j in range(seq_len[i]):
                if matrix[i][j] >= 2 * num_classes:
                    if matrix[i][j] == 4 * num_classes:
                        matrix[i][j] -= 2 * num_classes
                    elif matrix[i][j] >= 3 * num_classes:
                        matrix[i][j] -= 3 * num_classes
                    else:
                        matrix[i][j] -= num_classes

    b_labels = label_names[:num_classes]
    i_labels = label_names[num_classes*2:num_classes*3]
    label_names = b_labels + i_labels + [label_names[-1]]

    return gold, predicted, label_names


def compute_metrics(gold, predicted, seq_len, label_names):
    if are_iobes_labels(label_names):
        gold, predicted, label_names = iobes_to_iob(
            gold, predicted, label_names, seq_len
        )

    num_labels = len(label_names)

    conf = np.zeros([num_labels, num_labels], dtype=np.int32)
    for i in range(len(gold)):
        for j in range(seq_len[i]):
            conf[gold[i, j], predicted[i, j]] += 1

    acc = np.sum(np.diag(conf)) / sum(seq_len) * 100

    iob_labels = are_iob_labels(label_names)
    f1_required = f1_scores_required(gold, num_labels, seq_len)

    results = {
        "acc": acc, "confusion": conf,
        "F1": iob_labels or f1_required,
        "IOB": iob_labels
    }

    if results["F1"]:
        num_classes = num_labels // 2 if iob_labels else num_labels - 1

        b_tp = np.sum(conf[:num_classes, :num_classes])
        b_tn = np.sum(conf[num_classes:, num_classes:])
        b_fp = np.sum(conf[num_classes:, :num_classes])
        b_fn = np.sum(conf[:num_classes, num_classes:])
        b_prec, b_rec, b_f1 = compute_f1_score(b_tp, b_fp, b_fn)

        b_acc = np.sum(np.diag(conf[:num_classes])) / np.sum(conf[:num_classes]) * 100
        e_acc = np.sum(np.diag(conf[:num_labels-1])) / np.sum(conf[:num_labels-1]) * 100
        o_acc = conf[-1, -1] / np.sum(conf[-1]) * 100

        gold_entities, gold_spans = extract_entities(gold, seq_len, num_labels, num_classes)
        predicted_entities, predicted_spans = extract_entities(predicted, seq_len, num_labels, num_classes)

        e_tp = len([1 for p in predicted_spans if p in gold_spans])
        e_fp = len([1 for p in predicted_spans if p not in gold_spans])
        e_fn = len([1 for p in gold_spans if p not in predicted_spans])
        e_prec, e_rec, e_f1 = compute_f1_score(e_tp, e_fp, e_fn)

        ec_tp = len([1 for p in predicted_entities if p in gold_entities])
        ec_fp = len([1 for p in predicted_entities if p not in gold_entities])
        ec_fn = len([1 for p in gold_entities if p not in predicted_entities])
        ec_prec, ec_rec, ec_f1 = compute_f1_score(ec_tp, ec_fp, ec_fn)

        f1_metrics = {
            "B_acc": b_acc, "E_acc": e_acc, "O_acc": o_acc,
            "B_TP": b_tp, "B_TN": b_tn, "B_FP": b_fp, "B_FN": b_fn,
            "B_prec": b_prec, "B_rec": b_rec, "B_F1": b_f1,
            "E_TP": e_tp, "E_FP": e_fp, "E_FN": e_fn,
            "E_prec": e_prec, "E_rec": e_rec, "E_F1": e_f1,
            "EC_TP": ec_tp, "EC_FP": ec_fp, "EC_FN": ec_fn,
            "EC_prec": ec_prec, "EC_rec": ec_rec, "EC_F1": ec_f1,
            "CLASS": []
        }

        for i in range(num_classes):
            ec_class_tp = len([1 for p in predicted_entities if p[2] == i and p in gold_entities])
            ec_class_fp = len([1 for p in predicted_entities if p[2] == i and p not in gold_entities])
            ec_class_fn = len([1 for p in gold_entities if p[2] == i and p not in predicted_entities])
            ec_class_prec, ec_class_rec, ec_class_f1 = compute_f1_score(ec_class_tp, ec_class_fp, ec_class_fn)

            f1_metrics["CLASS"].append({
                "TP": ec_class_tp, "FP": ec_class_fp, "FN": ec_class_fn,
                "prec": ec_class_prec, "rec": ec_class_rec, "F1": ec_class_f1
            })

        for m in f1_metrics:
            results[m] = f1_metrics[m]

    return results


def get_performance_summary(metrics, num_labels):
    if metrics["F1"]:
        if metrics["IOB"]:
            return "{:<39} {:<30} {:<30} {}".format(
                "acc [B {d[B_acc]:.2f} E {d[E_acc]:.2f} O {d[O_acc]:.2f} T {d[acc]:.2f}]".format(d=metrics),
                "B [P {d[B_prec]:.2f} R {d[B_rec]:.2f} F1 {d[B_F1]:.2f}]".format(d=metrics),
                "E [P {d[E_prec]:.2f} R {d[E_rec]:.2f} F1 {d[E_F1]:.2f}]".format(d=metrics),
                "EC [P {d[EC_prec]:.2f} R {d[EC_rec]:.2f} F1 {d[EC_F1]:.2f}]".format(d=metrics)
            ), metrics["EC_F1"]
        else:
            if num_labels > 2:
                return "{:<30} {:<30} {}".format(
                    "acc [E {d[E_acc]:.2f} O {d[O_acc]:.2f} T {d[acc]:.2f}]".format(d=metrics),
                    "E [P {d[E_prec]:.2f} R {d[E_rec]:.2f} F1 {d[E_F1]:.2f}]".format(d=metrics),
                    "EC [P {d[EC_prec]:.2f} R {d[EC_rec]:.2f} F1 {d[EC_F1]:.2f}]".format(d=metrics)
                ), metrics["EC_F1"]
            else:
                return "{:<30} {:<30} {}".format(
                    "acc [E {d[E_acc]:.2f} O {d[O_acc]:.2f} T {d[acc]:.2f}]".format(d=metrics),
                    "E [TP {d[E_TP]} FP {d[E_FP]} FN {d[E_FN]}] ".format(d=metrics),
                    "[P {d[E_prec]:.2f} R {d[E_rec]:.2f} F1 {d[E_F1]:.2f}]".format(d=metrics)
                ), metrics["E_F1"]
    else:
        return "acc {d[acc]:.2f}".format(d=metrics), metrics["acc"]


def get_class_f1_summary(metrics, label_names):
    result = ""
    if "CLASS" in metrics and len(label_names) > 2:
        for i in range(len(metrics["CLASS"])):
            result += "{:<15} {:<25} {}\n".format(
                "{}".format(label_names[i][:2]),
                "[TP {d[TP]} FP {d[FP]} FN {d[FN]}]".format(d=metrics["CLASS"][i]),
                "[P {d[prec]:.2f} R {d[rec]:.2f} F1 {d[F1]:.2f}]".format(d=metrics["CLASS"][i])
            )

    return result


def visualize_predictions(sentences, gold, predicted, seq_len, label_names, num_samples=10):
    results = ""
    for i in range(min(len(sentences), num_samples)):
        visualized = "{}.".format(i+1)
        tokens = str(sentences[i], encoding="utf-8").split(" ")
        for j in range(seq_len[i]):
            gold_label = label_names[gold[i][j]]
            predicted_label = label_names[predicted[i][j]]
            visualized += " {}_{}".format(
                tokens[j],
                predicted_label if predicted_label == gold_label else "{}|{}".format(
                    predicted_label, gold_label
                )
            )
        results += visualized + "\n"
    return results
