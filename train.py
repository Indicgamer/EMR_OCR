import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

def run_training(model, processor, train_dataset, val_dataset, output_dir):
    # Load Character Error Rate metric
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        # Compute CER
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",        # <--- Change this from evaluation_strategy
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=True, 
    output_dir=output_dir,
    logging_steps=10,
    eval_steps=100,
    save_steps=100,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    return trainer