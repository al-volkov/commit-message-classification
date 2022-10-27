from src.train.model import Model

if __name__ == "__main__":
    model = Model()
    model.train()
    model.save_model("model.pth")
