document.addEventListener("DOMContentLoaded", function () {
  const button = document.getElementById("submit-btn");
  button.addEventListener("click", async (event) => {
    event.preventDefault();
    const ticker = document.getElementById("ticker").value;
    const data = await getPrediction(ticker);
  });
});

async function getPrediction(ticker) {
  clearGraph();
  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ticker: ticker }),
  });
    displayGraph(ticker);
}

function clearGraph() {
  const image_container = document.getElementById("graph-container")
  image_container.innerHTML = "";
}

function displayGraph(ticker) {
  
  const image_container = document.getElementById("graph-container")
  const img = document.createElement("img"); // Create an image element

  img.src = "static/" + ticker.toUpperCase() + ".png"; 
  img.width = 800;
  img.height = 600;

  image_container.appendChild(img);  
}
