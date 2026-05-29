def evaluate_model(model, test_gen):
    loss, acc = model.evaluate(test_gen)
    print("Test accuracy:", acc)
    return acc
