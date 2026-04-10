let session;
let preprocessing;

async function loadResources() {
  preprocessing = await fetch("preprocessing.json").then(res => res.json());

  const formDiv = document.getElementById("form");

  preprocessing.feature_names.forEach((name, index) => {
    const group = document.createElement("div");
    group.className = "input-group";

    const label = document.createElement("label");
    label.innerText = name;

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.id = "feature_" + index;

    group.appendChild(label);
    group.appendChild(input);
    formDiv.appendChild(group);
  });

  session = await ort.InferenceSession.create("breast_cancer_model.onnx");
}

async function predict() {
  const values = [];

  for (let i = 0; i < preprocessing.feature_names.length; i++) {
    let val = parseFloat(document.getElementById("feature_" + i).value);

    if (isNaN(val)) {
      alert("Please fill in all 30 fields.");
      return;
    }

    let scaled = (val - preprocessing.mean[i]) / preprocessing.scale[i];
    values.push(scaled);
  }

  const inputTensor = new ort.Tensor("float32", Float32Array.from(values), [1, 30]);

  const feeds = { input: inputTensor };
  const results = await session.run(feeds);

  const output = results.output.data;

  let prediction = output[0] > output[1] ? 0 : 1;

  const label = prediction === 0 ? "malignant" : "benign";
  document.getElementById("result").innerText = "Prediction: " + label;
}

loadResources();