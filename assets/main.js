const VISUALIZER_CONFIG = {
  weightUrl: "./exports/mlp_weights.json",
  maxConnectionsPerNeuron: 24,
  layerSpacing: 5.5,
  inputSpacing: 0.24,
  hiddenSpacing: 0.95,
  inputNodeSize: 0.18,
  hiddenNodeRadius: 0.22,
  connectionRadius: 0.005,
  connectionWeightThreshold: 0,
  showFpsOverlay: true,
  // When true the app uses a lightweight DOM-based fallback visualizer
  // instead of the Three.js 3D renderer. Useful for environments where
  // WebGL/three.js isn't available or causes deployment issues.
  useLegacyRenderer: false,
  brush: {
    drawRadius: 1.4,
    eraseRadius: 2.5,
    drawStrength: 0.95,
    eraseStrength: 0.95,
    softness: 0.3,
  },
};

const MNIST_SAMPLE_MANIFEST_URL = "./assets/data/mnist-test-manifest.json";

document.addEventListener("DOMContentLoaded", () => {
  initializeVisualizer().catch((error) => {
    console.error(error);
    renderErrorMessage("Visualization could not be initialized. Check the console for details.");
  });
});

async function loadMnistTestSamples(manifestPath = MNIST_SAMPLE_MANIFEST_URL) {
  const manifestUrl = new URL(manifestPath, window.location.href);
  const manifestResponse = await fetch(manifestUrl.toString());
  if (!manifestResponse.ok) {
    throw new Error(`Could not load MNIST manifest (${manifestResponse.status}).`);
  }
  const manifest = await manifestResponse.json();
  const rows = Number(manifest?.imageShape?.[0]) || 28;
  const cols = Number(manifest?.imageShape?.[1]) || 28;
  const numSamples = Number(manifest?.numSamples) || 0;
  const sampleSize = rows * cols;
  const imageFile = manifest?.image?.file;
  const labelFile = manifest?.labels?.file;
  if (!imageFile || !labelFile) {
    throw new Error("Manifest contains no valid file paths for images or labels.");
  }

  const [imageBuffer, labelBuffer] = await Promise.all([
    fetch(new URL(imageFile, manifestUrl).toString()).then((response) => {
      if (!response.ok) {
        throw new Error(`Could not load MNIST image data (${response.status}).`);
      }
      return response.arrayBuffer();
    }),
    fetch(new URL(labelFile, manifestUrl).toString()).then((response) => {
      if (!response.ok) {
        throw new Error(`Could not load MNIST label data (${response.status}).`);
      }
      return response.arrayBuffer();
    }),
  ]);

  const imageBytes = new Uint8Array(imageBuffer);
  const labelBytes = new Uint8Array(labelBuffer);
  if (numSamples <= 0) {
    if (sampleSize > 0) {
      const inferredSamples = Math.floor(imageBytes.length / sampleSize);
      if (inferredSamples <= 0) {
        throw new Error("Could not derive sample size from MNIST image data.");
      }
      if (labelBytes.length !== inferredSamples) {
        throw new Error("Number of labels does not match the derived samples.");
      }
    } else {
      throw new Error("Manifest does not contain a valid sample size.");
    }
  }

  const totalSamples = numSamples > 0 ? numSamples : Math.floor(imageBytes.length / sampleSize);
  if (imageBytes.length !== totalSamples * sampleSize) {
    throw new Error("MNIST image data length does not match expected size.");
  }
  if (labelBytes.length !== totalSamples) {
    throw new Error("MNIST label data length does not match expected size.");
  }

  const digitBuckets = Array.from({ length: 10 }, () => []);
  for (let index = 0; index < totalSamples; index += 1) {
    const digit = labelBytes[index];
    if (digitBuckets[digit]) {
      digitBuckets[digit].push(index);
    }
  }

  const pixelCache = new Map();
  const normaliseSlice = (index) => {
    if (pixelCache.has(index)) {
      return pixelCache.get(index);
    }
    const start = index * sampleSize;
    const slice = imageBytes.subarray(start, start + sampleSize);
    const normalized = new Float32Array(sampleSize);
    for (let i = 0; i < sampleSize; i += 1) {
      normalized[i] = slice[i] / 255;
    }
    pixelCache.set(index, normalized);
    return normalized;
  };

  return {
    rows,
    cols,
    sampleSize,
    totalSamples,
    getRandomSample(digit) {
      if (!Number.isInteger(digit) || digit < 0 || digit > 9) return null;
      const bucket = digitBuckets[digit];
      if (!bucket || bucket.length === 0) return null;
      const randomIndex = bucket[Math.floor(Math.random() * bucket.length)];
      return this.getSampleByIndex(randomIndex);
    },
    getSampleByIndex(index) {
      if (!Number.isFinite(index) || index < 0 || index >= totalSamples) return null;
      const pixels = normaliseSlice(index);
      return {
        index,
        digit: labelBytes[index],
        pixels,
      };
    },
  };
}

async function setupMnistSampleButtons({ digitCanvas, onSampleApplied, manifestPath } = {}) {
  if (!digitCanvas || typeof digitCanvas.setPixels !== "function") return null;
  const interactionRow =
    typeof digitCanvas.getInteractionRow === "function" ? digitCanvas.getInteractionRow() : null;
  const gridElement =
    typeof digitCanvas.getGridElement === "function" ? digitCanvas.getGridElement() : null;
  if (!interactionRow || !gridElement) return null;
  let loader;
  try {
    loader = await loadMnistTestSamples(manifestPath ?? MNIST_SAMPLE_MANIFEST_URL);
  } catch (error) {
    console.warn("Could not load MNIST test data:", error);
    return null;
  }
  const column = document.createElement("div");
  column.className = "digit-button-column";
  for (let digit = 0; digit < 10; digit += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "digit-button";
    button.textContent = String(digit);
    button.setAttribute("aria-label", `Load random ${digit}`);
    button.addEventListener("click", () => {
      const sample = loader.getRandomSample(digit);
      if (!sample) return;
      digitCanvas.setPixels(sample.pixels);
      if (typeof onSampleApplied === "function") {
        onSampleApplied(sample);
      }
  });
    column.appendChild(button);
  }
  interactionRow.appendChild(column);
  return {
    loader,
    column,
  };
}

async function initializeVisualizer() {
  initializeInfoDialog();

  const weightDefinitionUrl = new URL(VISUALIZER_CONFIG.weightUrl, window.location.href);
  const definition = await fetchNetworkDefinition(weightDefinitionUrl.toString());
  if (!definition?.network) {
    throw new Error("Invalid network definition.");
  }

  const timelineSnapshots = hydrateTimeline(definition.timeline, {
    layerMetadata: definition.network.layers,
    baseUrl: weightDefinitionUrl,
  });
  if (!timelineSnapshots.length) {
    throw new Error("No valid timeline snapshots found.");
  }
  const defaultSnapshotIndex = Math.max(timelineSnapshots.length - 1, 0);
  const initialSnapshot = timelineSnapshots[defaultSnapshotIndex];

  // Attempt to load snapshot layers; if snapshot files are missing or fail to load
  // create a minimal zero-initialized fallback from the network metadata so the
  // visualizer can still initialize instead of throwing a fatal error.
  let initialLayers;
  try {
    initialLayers = await initialSnapshot.loadLayers();
  } catch (err) {
    console.warn("Could not load snapshot layers, falling back to zeroed layers:", err);
    const metaLayers = Array.isArray(definition.network.layers)
      ? definition.network.layers
      : [];
    initialLayers = metaLayers.map((layerMeta, idx) => {
      const weightShape = Array.isArray(layerMeta?.weight_shape) ? layerMeta.weight_shape.slice() : [];
      const biasShape = Array.isArray(layerMeta?.bias_shape) ? layerMeta.bias_shape.slice() : [];
      const rows = Math.max(0, Number(weightShape[0]) || 0);
      const cols = Math.max(0, Number(weightShape[1]) || 0);
      const biasLen = Math.max(0, Number(biasShape[0]) || rows);
      const weights = [];
      for (let r = 0; r < rows; r += 1) {
        const row = new Float32Array(cols);
        weights.push(row);
      }
      const biases = new Float32Array(biasLen);
      return {
        name: typeof layerMeta?.name === "string" ? layerMeta.name : `dense_${idx}`,
        activation: typeof layerMeta?.activation === "string" ? layerMeta.activation : "relu",
        weights,
        biases,
      };
    });
  }

  const neuralModel = new FeedForwardModel({
    normalization: definition.network.normalization,
    architecture: definition.network.architecture,
    layers: initialLayers,
  });
  const gridContainerElement = document.getElementById("gridContainer");
  const digitCanvas = new DigitSketchPad(gridContainerElement, 28, 28, {
    brush: VISUALIZER_CONFIG.brush,
  });
  const probabilityPanel = new ProbabilityPanel(document.getElementById("predictionChart"));
  const networkInfoPanelElement = document.getElementById("networkInfoPanel");
  const networkInfoPanel = networkInfoPanelElement ? new NetworkInfoPanel(networkInfoPanelElement) : null;
  const neuronDetailPanelElement = document.getElementById("neuronDetailPanel");
  const neuronDetailPanel = new NeuronDetailPanel(neuronDetailPanelElement);
  // Create either the WebGL-based visualizer or a lightweight DOM fallback
  // depending on `VISUALIZER_CONFIG.useLegacyRenderer`.
  let sceneController = null;
  if (VISUALIZER_CONFIG.useLegacyRenderer) {
    sceneController = new LegacyVisualizer(neuralModel, {
      layerSpacing: VISUALIZER_CONFIG.layerSpacing,
      maxConnectionsPerNeuron: VISUALIZER_CONFIG.maxConnectionsPerNeuron,
      inputSpacing: VISUALIZER_CONFIG.inputSpacing,
      hiddenSpacing: VISUALIZER_CONFIG.hiddenSpacing,
      inputNodeSize: VISUALIZER_CONFIG.inputNodeSize,
      hiddenNodeRadius: VISUALIZER_CONFIG.hiddenNodeRadius,
      connectionRadius: VISUALIZER_CONFIG.connectionRadius,
      connectionWeightThreshold: VISUALIZER_CONFIG.connectionWeightThreshold,
    });
  } else {
    sceneController = new NeuralVisualizer(neuralModel, {
      layerSpacing: VISUALIZER_CONFIG.layerSpacing,
      maxConnectionsPerNeuron: VISUALIZER_CONFIG.maxConnectionsPerNeuron,
      inputSpacing: VISUALIZER_CONFIG.inputSpacing,
      hiddenSpacing: VISUALIZER_CONFIG.hiddenSpacing,
      inputNodeSize: VISUALIZER_CONFIG.inputNodeSize,
      hiddenNodeRadius: VISUALIZER_CONFIG.hiddenNodeRadius,
      connectionRadius: VISUALIZER_CONFIG.connectionRadius,
      connectionWeightThreshold: VISUALIZER_CONFIG.connectionWeightThreshold,
      showFpsOverlay: VISUALIZER_CONFIG.showFpsOverlay,
      onNeuronFocusChange: (payload) => neuronDetailPanel.update(payload),
    });
  }
  networkInfoPanel?.update(neuralModel);
  neuronDetailPanel.setOnClear(() => {
    if (sceneController && typeof sceneController.clearSelection === "function") {
      sceneController.clearSelection();
    }
  });

  if (gridContainerElement && neuronDetailPanelElement) {
    const rootStyle = document.documentElement?.style ?? null;
    const spacingBelowSketchPad = 16;
    const bottomMargin = 24;

    const applyNeuronPanelLayout = () => {
      if (!rootStyle) return;
      const gridRect = gridContainerElement.getBoundingClientRect();
      const rawTop = Math.round(gridRect.bottom + spacingBelowSketchPad);
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
      const availableHeight = Math.max(viewportHeight - rawTop - bottomMargin, 200);
      rootStyle.setProperty("--neuron-panel-top", `${rawTop}px`);
      rootStyle.setProperty("--neuron-panel-max-height", `${availableHeight}px`);
    };

    const scheduleNeuronPanelLayout = () => window.requestAnimationFrame(applyNeuronPanelLayout);
    scheduleNeuronPanelLayout();
    window.addEventListener("resize", scheduleNeuronPanelLayout, { passive: true });

    if (typeof ResizeObserver !== "undefined") {
      if (neuronDetailPanelElement.__gridResizeObserver instanceof ResizeObserver) {
        neuronDetailPanelElement.__gridResizeObserver.disconnect();
      }
      const gridResizeObserver = new ResizeObserver(() => scheduleNeuronPanelLayout());
      gridResizeObserver.observe(gridContainerElement);
      neuronDetailPanelElement.__gridResizeObserver = gridResizeObserver;
    }
  }

  if (typeof window !== "undefined") {
    // Expose active scene controller instance for interactive inspection in DevTools.
    try {
      window.sceneController = sceneController;
    } catch (_e) {
      // ignore
    }
  }

  const resetBtn = document.getElementById("resetBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      digitCanvas.clear();
      refreshNetworkState();
    });
  }

  function refreshNetworkState() {
    const rawInput = digitCanvas.getPixels();
    const propagation = neuralModel.propagate(rawInput);
    const displayActivations = propagation.activations.slice();
    if (displayActivations.length > 0) {
      displayActivations[0] = rawInput;
    }

    const logitsTyped =
      propagation.preActivations.length > 0
        ? propagation.preActivations[propagation.preActivations.length - 1]
        : new Float32Array(0);
    const probabilities =
      logitsTyped.length > 0 ? Float32Array.from(softmax(Array.from(logitsTyped))) : new Float32Array(0);

    if (probabilities.length && displayActivations.length > 1) {
      displayActivations[displayActivations.length - 1] = probabilities;
    }

    let networkActivations = propagation.activations;
    if (probabilities.length) {
      networkActivations = propagation.activations.slice();
      if (networkActivations.length > 1) {
        networkActivations[networkActivations.length - 1] = probabilities;
      }
    }

    if (sceneController && typeof sceneController.update === "function") {
      sceneController.update(displayActivations, networkActivations, propagation.preActivations);
    }
    const probabilitiesForPanel = probabilities.length ? probabilities : logitsTyped;
    probabilityPanel.update(probabilitiesForPanel.length ? Array.from(probabilitiesForPanel) : []);

    // No noisy debug logs; visual updates are visible in-scene
  }

  await setupMnistSampleButtons({
    digitCanvas,
    onSampleApplied: () => refreshNetworkState(),
  });

  initializeAdvancedSettings({
    sceneController,
    digitCanvas,
    onConnectionsSettingsChange() {
      refreshNetworkState();
    },
  });

  const timelineController = setupTimelineSlider(timelineSnapshots, {
    async onSnapshotChange(snapshot) {
      if (!snapshot) return;
      let layers;
      try {
        layers = await snapshot.loadLayers();
      } catch (err) {
        console.warn("Failed to load snapshot, using fallback zeroed layers:", err);
        const metaLayers = Array.isArray(definition.network.layers) ? definition.network.layers : [];
        layers = metaLayers.map((layerMeta, idx) => {
          const weightShape = Array.isArray(layerMeta?.weight_shape) ? layerMeta.weight_shape.slice() : [];
          const biasShape = Array.isArray(layerMeta?.bias_shape) ? layerMeta.bias_shape.slice() : [];
          const rows = Math.max(0, Number(weightShape[0]) || 0);
          const cols = Math.max(0, Number(weightShape[1]) || 0);
          const biasLen = Math.max(0, Number(biasShape[0]) || rows);
          const weights = [];
          for (let r = 0; r < rows; r += 1) {
            weights.push(new Float32Array(cols));
          }
          const biases = new Float32Array(biasLen);
          return {
            name: typeof layerMeta?.name === "string" ? layerMeta.name : `dense_${idx}`,
            activation: typeof layerMeta?.activation === "string" ? layerMeta.activation : "relu",
            weights,
            biases,
          };
        });
      }
      neuralModel.updateLayers(layers);
      if (sceneController && typeof sceneController.updateNetworkWeights === "function") {
        sceneController.updateNetworkWeights();
      }
      networkInfoPanel?.update(neuralModel);
      refreshNetworkState();
    },
  });

  digitCanvas.setChangeHandler(() => refreshNetworkState());

  if (timelineController && typeof timelineController.setActiveIndex === "function") {
    await timelineController.setActiveIndex(defaultSnapshotIndex, { emit: true, force: true });
  } else {
    networkInfoPanel?.update(neuralModel);
    refreshNetworkState();
  }
}

function initializeInfoDialog() {
  const infoButton = document.getElementById("infoButton");
  const infoModal = document.getElementById("infoModal");
  const closeButton = document.getElementById("closeInfoModal");
  if (!infoModal) return;

  const showModal = () => infoModal.classList.add("visible");
  const hideModal = () => infoModal.classList.remove("visible");

  infoButton?.addEventListener("click", showModal);
  closeButton?.addEventListener("click", hideModal);
  infoModal.addEventListener("click", (event) => {
    if (event.target === infoModal) {
      hideModal();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && infoModal.classList.contains("visible")) {
      hideModal();
    }
  });
}

function initializeAdvancedSettings({ sceneController, digitCanvas, onConnectionsSettingsChange } = {}) {
  const button = document.getElementById("advancedSettingsButton");
  const modal = document.getElementById("advancedSettingsModal");
  const closeButton = document.getElementById("closeAdvancedSettings");
  if (!button || !modal) return;

  const connectionSlider = document.getElementById("connectionLimitSlider");
  const connectionValue = document.getElementById("connectionLimitValue");
  const connectionThresholdSlider = document.getElementById("connectionThresholdSlider");
  const connectionThresholdValue = document.getElementById("connectionThresholdValue");
  const connectionThicknessSlider = document.getElementById("connectionThicknessSlider");
  const connectionThicknessValue = document.getElementById("connectionThicknessValue");
  const thicknessSlider = document.getElementById("brushThicknessSlider");
  const thicknessValue = document.getElementById("brushThicknessValue");
  const strengthSlider = document.getElementById("brushStrengthSlider");
  const strengthValue = document.getElementById("brushStrengthValue");
  let refreshConnectionThresholdBounds = null;

  const focusTarget =
    connectionSlider ||
    connectionThresholdSlider ||
    connectionThicknessSlider ||
    thicknessSlider ||
    strengthSlider;

  const showModal = () => {
    modal.classList.add("visible");
    if (typeof refreshConnectionThresholdBounds === "function") {
      refreshConnectionThresholdBounds();
    }
    window.requestAnimationFrame(() => {
      if (focusTarget && typeof focusTarget.focus === "function") {
        try {
          focusTarget.focus({ preventScroll: true });
        } catch (_error) {
          focusTarget.focus();
        }
      }
    });
  };

  const hideModal = () => {
    modal.classList.remove("visible");
    if (typeof button.focus === "function") {
      try {
        button.focus({ preventScroll: true });
      } catch (_error) {
        button.focus();
      }
    }
  };

  button.addEventListener("click", showModal);
  closeButton?.addEventListener("click", hideModal);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      hideModal();
    }
  });

  if (connectionSlider && connectionValue && sceneController) {
    let maxIncoming = 0;
    if (Array.isArray(sceneController.mlp?.layers)) {
      for (const layer of sceneController.mlp.layers) {
        if (!layer || !Array.isArray(layer.weights) || !layer.weights.length) continue;
        const rowLength = layer.weights[0]?.length ?? 0;
        if (rowLength > maxIncoming) {
          maxIncoming = rowLength;
        }
      }
    }
    const sliderMax = Math.max(1, maxIncoming || Number.parseInt(connectionSlider.max, 10) || 64);
    connectionSlider.max = String(sliderMax);

    const syncConnectionUi = (value) => {
      const normalized = Number.isFinite(value) ? value : sceneController.options.maxConnectionsPerNeuron;
      connectionSlider.value = String(normalized);
      connectionValue.textContent = `${normalized}`;
    };

    const applyConnectionLimit = (rawValue, { emit = true } = {}) => {
      let parsed = Number.parseInt(rawValue, 10);
      if (!Number.isFinite(parsed)) {
        parsed = sceneController.options.maxConnectionsPerNeuron;
      }
      const maxValue = Number.parseInt(connectionSlider.max, 10) || sliderMax;
      const clamped = Math.min(maxValue, Math.max(1, parsed));
      syncConnectionUi(clamped);
      if (!emit) return;
      const changed = sceneController.setMaxConnectionsPerNeuron(clamped);
      if (changed) {
        if (typeof onConnectionsSettingsChange === "function") {
          onConnectionsSettingsChange(clamped);
        }
        VISUALIZER_CONFIG.maxConnectionsPerNeuron = clamped;
      }
    };

    applyConnectionLimit(sceneController.options.maxConnectionsPerNeuron, { emit: false });

    connectionSlider.addEventListener("input", (event) => {
      applyConnectionLimit(event.target.value);
    });

    connectionSlider.addEventListener("change", (event) => {
      applyConnectionLimit(event.target.value);
    });
  } else if (connectionSlider) {
    connectionSlider.disabled = true;
    if (connectionValue) {
      connectionValue.textContent = "—";
    }
  }

  if (connectionThresholdSlider && connectionThresholdValue && sceneController) {
    const min = Math.max(0, Number.parseFloat(connectionThresholdSlider.min) || 0);
    const fallbackMaxAttr = Number.parseFloat(connectionThresholdSlider.getAttribute("max"));
    const computeSceneMaxMagnitude = () => {
      if (typeof sceneController.getMaxConnectionWeightMagnitude === "function") {
        const sceneValue = sceneController.getMaxConnectionWeightMagnitude();
        if (Number.isFinite(sceneValue)) {
          return sceneValue;
        }
      }
      let maxMagnitude = 0;
      if (Array.isArray(sceneController.mlp?.layers)) {
        for (const layer of sceneController.mlp.layers) {
          if (!Array.isArray(layer?.weights)) continue;
          for (const row of layer.weights) {
            if (!Array.isArray(row)) continue;
            for (const weight of row) {
              const magnitude = Math.abs(Number(weight));
              if (Number.isFinite(magnitude) && magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
              }
            }
          }
        }
      }
      return maxMagnitude;
    };
    const updateThresholdSliderBounds = () => {
      const sceneMax = computeSceneMaxMagnitude();
      const fallback = Number.isFinite(fallbackMaxAttr) ? fallbackMaxAttr : 0;
      const resolved = Math.max(sceneMax || 0, fallback, 0.01);
      connectionThresholdSlider.max = String(resolved);
      return resolved;
    };
    const formatThreshold = (value) => {
      if (!Number.isFinite(value)) return "0.0000";
      if (value >= 1) return value.toFixed(2);
      if (value >= 0.1) return value.toFixed(3);
      if (value >= 0.01) return value.toFixed(4);
      return value.toFixed(5);
    };
    const syncThresholdUi = (value) => {
      connectionThresholdSlider.value = String(value);
      connectionThresholdValue.textContent = formatThreshold(value);
    };
    const applyConnectionThreshold = (rawValue, { emit = true } = {}) => {
      const max = updateThresholdSliderBounds();
      let parsed = Number.parseFloat(rawValue);
      if (!Number.isFinite(parsed)) {
        parsed = sceneController.options.connectionWeightThreshold ?? 0;
      }
      const clamped = clamp(parsed, min, max);
      syncThresholdUi(clamped);
      if (!emit) return;
      const changed = sceneController.setConnectionWeightThreshold(clamped);
      if (changed) {
        VISUALIZER_CONFIG.connectionWeightThreshold = clamped;
        if (typeof onConnectionsSettingsChange === "function") {
          onConnectionsSettingsChange(clamped);
        }
      }
    };
    const initialThreshold = Number.isFinite(sceneController.options.connectionWeightThreshold)
      ? Math.max(min, sceneController.options.connectionWeightThreshold)
      : Math.max(min, VISUALIZER_CONFIG.connectionWeightThreshold);
    updateThresholdSliderBounds();
    applyConnectionThreshold(initialThreshold, { emit: false });

    connectionThresholdSlider.addEventListener("input", (event) => {
      applyConnectionThreshold(event.target.value);
    });
    connectionThresholdSlider.addEventListener("change", (event) => {
      applyConnectionThreshold(event.target.value);
    });

    refreshConnectionThresholdBounds = () => {
      applyConnectionThreshold(connectionThresholdSlider.value, { emit: false });
    };
  } else if (connectionThresholdSlider) {
    connectionThresholdSlider.disabled = true;
    if (connectionThresholdValue) {
      connectionThresholdValue.textContent = "—";
    }
  }

  if (connectionThicknessSlider && connectionThicknessValue && sceneController) {
    const min = Number.parseFloat(connectionThicknessSlider.min) || 0.001;
    const max = Number.parseFloat(connectionThicknessSlider.max) || 0.1;
    const formatThickness = (value) => `${value.toFixed(3)}`;
    const syncConnectionThicknessUi = (value) => {
      connectionThicknessSlider.value = String(value);
      connectionThicknessValue.textContent = formatThickness(value);
    };
    const applyConnectionThickness = (rawValue, { emit = true } = {}) => {
      let parsed = Number.parseFloat(rawValue);
      if (!Number.isFinite(parsed)) {
        parsed = sceneController.options.connectionRadius;
      }
      const clamped = clamp(parsed, min, max);
      syncConnectionThicknessUi(clamped);
      if (!emit) return;
      const changed = sceneController.setConnectionRadius(clamped);
      if (changed) {
        VISUALIZER_CONFIG.connectionRadius = clamped;
      }
    };
    const initialRadius = Number.isFinite(sceneController.options.connectionRadius)
      ? clamp(sceneController.options.connectionRadius, min, max)
      : clamp(VISUALIZER_CONFIG.connectionRadius, min, max);
    applyConnectionThickness(initialRadius, { emit: false });

    connectionThicknessSlider.addEventListener("input", (event) => {
      applyConnectionThickness(event.target.value);
    });
    connectionThicknessSlider.addEventListener("change", (event) => {
      applyConnectionThickness(event.target.value);
    });
  } else if (connectionThicknessSlider) {
    connectionThicknessSlider.disabled = true;
    if (connectionThicknessValue) {
      connectionThicknessValue.textContent = "—";
    }
  }

  if (digitCanvas && typeof digitCanvas.getBrushSettings === "function") {
    const brush = digitCanvas.getBrushSettings();
    if (thicknessSlider && thicknessValue) {
      const min = Number.parseFloat(thicknessSlider.min) || 0.1;
      const max = Number.parseFloat(thicknessSlider.max) || 6;
      const formatThickness = (value) => value.toFixed(1);
      const syncThicknessUi = (value) => {
        thicknessSlider.value = String(value);
        thicknessValue.textContent = formatThickness(value);
      };
      const applyThickness = (rawValue) => {
        let parsed = Number.parseFloat(rawValue);
        if (!Number.isFinite(parsed)) {
          parsed = brush.drawRadius;
        }
        const clamped = clamp(parsed, min, max);
        syncThicknessUi(clamped);
        digitCanvas.updateBrushSettings({ drawRadius: clamped });
        VISUALIZER_CONFIG.brush.drawRadius = clamped;
        brush.drawRadius = clamped;
      };
      applyThickness(brush.drawRadius);
      thicknessSlider.addEventListener("input", (event) => {
        applyThickness(event.target.value);
      });
      thicknessSlider.addEventListener("change", (event) => {
        applyThickness(event.target.value);
      });
    }

    if (strengthSlider && strengthValue) {
      const min = Number.parseFloat(strengthSlider.min) || 0;
      const max = Number.parseFloat(strengthSlider.max) || 1;
      const formatStrength = (value) => `${Math.round(value * 100)}%`;
      const syncStrengthUi = (value) => {
        strengthSlider.value = String(value);
        strengthValue.textContent = formatStrength(value);
      };
      const applyStrength = (rawValue) => {
        let parsed = Number.parseFloat(rawValue);
        if (!Number.isFinite(parsed)) {
          parsed = brush.drawStrength;
        }
        const clamped = clamp(parsed, min, max);
        syncStrengthUi(clamped);
        digitCanvas.updateBrushSettings({ drawStrength: clamped });
        VISUALIZER_CONFIG.brush.drawStrength = clamped;
        brush.drawStrength = clamped;
      };
      applyStrength(brush.drawStrength);
      strengthSlider.addEventListener("input", (event) => {
        applyStrength(event.target.value);
      });
      strengthSlider.addEventListener("change", (event) => {
        applyStrength(event.target.value);
      });
    }
  } else {
    if (thicknessSlider) thicknessSlider.disabled = true;
    if (thicknessValue) thicknessValue.textContent = "—";
    if (strengthSlider) strengthSlider.disabled = true;
    if (strengthValue) strengthValue.textContent = "—";
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && modal.classList.contains("visible")) {
      hideModal();
    }
  });
}

async function fetchNetworkDefinition(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load network weights (${response.status})`);
  }
  return response.json();
}

function renderErrorMessage(message) {
  const chart = document.getElementById("predictionChart");
  if (chart) {
    chart.innerHTML = `<p class="error-text">${message}</p>`;
  }
}

function resolveRelativeUrl(base, relativePath) {
  try {
    const baseUrl = base instanceof URL ? base : new URL(base, window.location.href);
    return new URL(relativePath, baseUrl).toString();
  } catch (error) {
    console.warn("Could not resolve relative URL:", relativePath, error);
    return null;
  }
}

function decodeBase64ToUint8Array(base64) {
  if (typeof atob === "function") {
    const binary = atob(base64);
    const length = binary.length;
    const bytes = new Uint8Array(length);
    for (let i = 0; i < length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }
  if (typeof Buffer === "function") {
    return Uint8Array.from(Buffer.from(base64, "base64"));
  }
  throw new Error("Base64 decoding is not available in this environment.");
}

function float16ToFloat32(value) {
  const sign = (value & 0x8000) >> 15;
  const exponent = (value & 0x7c00) >> 10;
  const fraction = value & 0x03ff;

  let result;
  if (exponent === 0) {
    if (fraction === 0) {
      result = 0;
    } else {
      result = (fraction / 0x400) * Math.pow(2, -14);
    }
  } else if (exponent === 0x1f) {
    result = fraction === 0 ? Number.POSITIVE_INFINITY : Number.NaN;
  } else {
    result = (1 + fraction / 0x400) * Math.pow(2, exponent - 15);
  }

  return sign === 1 ? -result : result;
}

function decodeFloat16Base64(base64, expectedLength) {
  const bytes = decodeBase64ToUint8Array(base64);
  if (bytes.byteLength % 2 !== 0) {
    throw new Error("Float16 data has an invalid length.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const length = bytes.byteLength / 2;
  if (Number.isFinite(expectedLength) && expectedLength > 0 && length !== expectedLength) {
    throw new Error(
      `Expected ${expectedLength} Float16 values, but received ${length}.`,
    );
  }
  const result = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const half = view.getUint16(index * 2, true);
    result[index] = float16ToFloat32(half);
  }
  return result;
}

function decodeWeightMatrix(encoded, shape) {
  const rows = Math.max(0, Number(shape?.[0]) || 0);
  const cols = Math.max(0, Number(shape?.[1]) || 0);
  if (rows === 0 || cols === 0) {
    return [];
  }
  const flat = decodeFloat16Base64(encoded, rows * cols);
  const result = [];
  for (let row = 0; row < rows; row += 1) {
    const start = row * cols;
    const end = start + cols;
    result.push(flat.slice(start, end));
  }
  return result;
}

function normaliseShape(shape, fallback = []) {
  const source = Array.isArray(shape) ? shape : fallback;
  if (!Array.isArray(source)) return [];
  return source.map((value) => Number(value) || 0);
}

function normaliseLayerMetadata(layer, index) {
  const layerIndex = Number.isFinite(layer?.layer_index) ? Number(layer.layer_index) : index;
  const weightShape = normaliseShape(layer?.weight_shape);
  const biasShape = normaliseShape(layer?.bias_shape);
  const resolvedWeightShape =
    weightShape.length === 2 ? weightShape : [biasShape[0] ?? 0, weightShape[1] ?? 0];
  const resolvedBiasShape = biasShape.length >= 1 ? biasShape : [resolvedWeightShape[0] ?? 0];
  return {
    layerIndex,
    name: typeof layer?.name === "string" ? layer.name : `dense_${layerIndex}`,
    activation: typeof layer?.activation === "string" ? layer.activation : "relu",
    weightShape: resolvedWeightShape,
    biasShape: resolvedBiasShape,
  };
}

function normaliseWeightsDescriptor(descriptor, baseUrl) {
  if (!descriptor || typeof descriptor !== "object") return null;
  const path = typeof descriptor.path === "string" ? descriptor.path : null;
  if (!path) return null;
  const url = resolveRelativeUrl(baseUrl, path);
  if (!url) return null;
  return {
    path,
    url,
    dtype: typeof descriptor.dtype === "string" ? descriptor.dtype : "float16",
    format: typeof descriptor.format === "string" ? descriptor.format : "layer_array_v1",
  };
}

async function fetchSnapshotPayload(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load snapshot (${response.status})`);
  }
  return response.json();
}

function decodeSnapshotLayers(payload, layerMetadata) {
  if (!payload || typeof payload !== "object" || !Array.isArray(payload.layers)) {
    throw new Error("Snapshot file contains no valid layer data.");
  }

  return layerMetadata.map((meta, index) => {
    const layerPayload =
      payload.layers[index] ??
      payload.layers.find((layer) => Number(layer?.layer_index) === meta.layerIndex);
    if (!layerPayload) {
      throw new Error(`Snapshot is missing layer ${meta.layerIndex}.`);
    }
    const weightsInfo = layerPayload.weights ?? {};
    const biasesInfo = layerPayload.biases ?? {};
    if (typeof weightsInfo.data !== "string" || typeof biasesInfo.data !== "string") {
      throw new Error("Snapshot layer contains no encoded weights.");
    }

    const weightShape = normaliseShape(weightsInfo.shape, meta.weightShape);
    const biasShape = normaliseShape(biasesInfo.shape, meta.biasShape);
    if (weightShape.length !== 2) {
      throw new Error("Snapshot layer has an invalid weight dimension.");
    }
    if (biasShape.length === 0) {
      throw new Error("Snapshot layer has an invalid bias dimension.");
    }

    const weights = decodeWeightMatrix(weightsInfo.data, weightShape);
    const biases = decodeFloat16Base64(biasesInfo.data, biasShape[0]);
    return {
      name: typeof layerPayload.name === "string" ? layerPayload.name : meta.name,
      activation:
        typeof layerPayload.activation === "string" ? layerPayload.activation : meta.activation,
      weights,
      biases,
    };
  });
}

function hydrateTimeline(rawTimeline, options = {}) {
  if (!Array.isArray(rawTimeline)) return [];

  const layerMetadataSource = Array.isArray(options.layerMetadata) ? options.layerMetadata : [];
  if (layerMetadataSource.length === 0) return [];
  const layerMetadata = layerMetadataSource.map((layer, index) =>
    normaliseLayerMetadata(layer, index),
  );

  const baseUrl = options.baseUrl ?? window.location.href;

  return rawTimeline
    .map((entry, index) => {
      if (!entry || typeof entry !== "object") return null;

      const weights = normaliseWeightsDescriptor(entry.weights, baseUrl);
      if (!weights?.url) return null;

      const metrics = typeof entry.metrics === "object" && entry.metrics !== null ? entry.metrics : {};
      const snapshot = {
        id: typeof entry.id === "string" ? entry.id : `snapshot_${index}`,
        order: Number.isFinite(entry.order) ? Number(entry.order) : index,
        label: typeof entry.label === "string" ? entry.label : `Snapshot ${index + 1}`,
        description: typeof entry.description === "string" ? entry.description : "",
        kind: typeof entry.kind === "string" ? entry.kind : "approx",
        imagesSeen: Number.isFinite(entry.images_seen) ? entry.images_seen : null,
        targetImages: Number.isFinite(entry.target_images) ? entry.target_images : null,
        batchesSeen: Number.isFinite(entry.batches_seen) ? entry.batches_seen : null,
        datasetPasses: Number.isFinite(entry.dataset_passes) ? entry.dataset_passes : null,
        datasetMultiple: Number.isFinite(entry.dataset_multiple) ? entry.dataset_multiple : null,
        metrics: {
          testAccuracy: Number.isFinite(metrics.test_accuracy) ? metrics.test_accuracy : null,
          avgTrainingLoss: Number.isFinite(metrics.avg_training_loss) ? metrics.avg_training_loss : null,
        },
        weights,
        layers: null,
        async loadLayers() {
          if (Array.isArray(this.layers) && this.layers.length) {
            return this.layers;
          }
          const payload = await fetchSnapshotPayload(this.weights.url);
          this.layers = decodeSnapshotLayers(payload, layerMetadata);
          return this.layers;
        },
      };
      return snapshot;
    })
    .filter(Boolean);
}

function formatInteger(value) {
  if (!Number.isFinite(value)) return "";
  return Math.round(value).toLocaleString();
}

function formatDecimal(value, digits) {
  if (!Number.isFinite(value)) return "";
  return value.toFixed(digits);
}

function formatSnapshotDescription(snapshot) {
  if (snapshot.description) return snapshot.description;
  const parts = [];
  if (Number.isFinite(snapshot.imagesSeen)) {
    parts.push(`${formatInteger(snapshot.imagesSeen)} images`);
  }
  if (Number.isFinite(snapshot.datasetMultiple)) {
    parts.push(`${snapshot.datasetMultiple}× dataset`);
  } else if (Number.isFinite(snapshot.datasetPasses)) {
    parts.push(`${formatDecimal(snapshot.datasetPasses, 2)}× dataset`);
  }
  if (Number.isFinite(snapshot.batchesSeen)) {
    parts.push(`${formatInteger(snapshot.batchesSeen)} batches`);
  }
  return parts.join(" • ");
}

function formatTimelineMetrics(metrics) {
  if (!metrics) return "";
  const segments = [];
  if (Number.isFinite(metrics.testAccuracy)) {
    segments.push(`Test acc: ${formatDecimal(metrics.testAccuracy * 100, 2)}%`);
  }
  if (Number.isFinite(metrics.avgTrainingLoss)) {
    segments.push(`Avg loss: ${formatDecimal(metrics.avgTrainingLoss, 4)}`);
  }
  return segments.join(" • ");
}

function setupTimelineSlider(timelineSnapshots, options = {}) {
  const overlay = document.getElementById("timelineOverlay");
  const slider = document.getElementById("timelineSlider");
  const summary = document.getElementById("timelineSummary");
  const label = document.getElementById("timelineLabel");
  const metricsElement = document.getElementById("timelineMetrics");

  if (
    !overlay ||
    !slider ||
    !summary ||
    !label ||
    !metricsElement ||
    !Array.isArray(timelineSnapshots) ||
    timelineSnapshots.length === 0
  ) {
    overlay?.classList.add("hidden");
    if (slider) {
      slider.disabled = true;
      slider.value = "0";
    }
    if (summary) summary.textContent = "";
    if (label) label.textContent = "";
    if (metricsElement) metricsElement.textContent = "";
    return null;
  }

  overlay.classList.remove("hidden");
  slider.min = "0";
  slider.max = String(Math.max(timelineSnapshots.length - 1, 0));
  slider.step = "1";
  slider.disabled = timelineSnapshots.length <= 1;

  const state = {
    activeIndex: null,
    loading: false,
  };

  const setLoading = (value) => {
    state.loading = Boolean(value);
    if (state.loading) {
      slider.disabled = true;
      overlay.classList.add("timeline-overlay--loading");
    } else {
      slider.disabled = timelineSnapshots.length <= 1;
      overlay.classList.remove("timeline-overlay--loading");
    }
  };

  const applySnapshotChange = async (snapshot, safeIndex) => {
    if (typeof options.onSnapshotChange !== "function") return;
    setLoading(true);
    try {
      await options.onSnapshotChange(snapshot, safeIndex);
    } finally {
      setLoading(false);
    }
  };

  const setActiveIndex = async (index, { emit = false, force = false } = {}) => {
    if (!Number.isFinite(index)) return null;
    const safeIndex = Math.round(index);
    if (safeIndex < 0 || safeIndex >= timelineSnapshots.length) return null;
    if (!force && state.activeIndex === safeIndex) return timelineSnapshots[safeIndex];
    if (state.loading && !force) return null;

    state.activeIndex = safeIndex;
    slider.value = String(safeIndex);

    const snapshot = timelineSnapshots[safeIndex];
    summary.textContent = snapshot.label;
    label.textContent = formatSnapshotDescription(snapshot);

    const metricsText = formatTimelineMetrics(snapshot.metrics);
    metricsElement.textContent = metricsText || "";
    metricsElement.classList.toggle("timeline-metrics--empty", metricsText.length === 0);

    if (emit) {
      await applySnapshotChange(snapshot, safeIndex);
    }

    return snapshot;
  };

  slider.addEventListener("input", (event) => {
    const nextIndex = Number(event.target.value);
    if (Number.isNaN(nextIndex)) return;
    setActiveIndex(nextIndex, { emit: true }).catch((error) => {
      console.error("Fehler beim Aktualisieren des Snapshots:", error);
    });
  });

  slider.addEventListener("change", (event) => {
    const nextIndex = Number(event.target.value);
    if (Number.isNaN(nextIndex)) return;
    setActiveIndex(nextIndex, { emit: true }).catch((error) => {
      console.error("Fehler beim Aktualisieren des Snapshots:", error);
    });
  });

  return {
    setActiveIndex,
    get activeIndex() {
      return state.activeIndex;
    },
    get loading() {
      return state.loading;
    },
  };
}

class DigitSketchPad {
  constructor(container, rows, cols, options = {}) {
    if (!container) {
      throw new Error("Grid container not found.");
    }
    this.container = container;
    this.rows = rows;
    this.cols = cols;
    this.values = new Float32Array(rows * cols);
    this.cells = [];
    this.isDrawing = false;
    this.activeMode = "draw";
    this.onChange = null;
    this.pendingChange = false;
    this.interactionRow = null;
    const defaultBrush = {
      drawRadius: 1.2,
      eraseRadius: 1.2,
      drawStrength: 0.85,
      eraseStrength: 0.8,
      softness: 0.5,
    };
    this.brush = Object.assign(defaultBrush, options.brush || {});
    this.buildGrid();
  }

  buildGrid() {
    this.gridElement = document.createElement("div");
    this.gridElement.className = "grid";
    this.gridElement.style.gridTemplateColumns = `repeat(${this.cols}, 1fr)`;
    this.gridElement.style.gridTemplateRows = `repeat(${this.rows}, 1fr)`;

    for (let i = 0; i < this.values.length; i += 1) {
      const cell = document.createElement("div");
      cell.className = "grid-cell";
      cell.dataset.index = String(i);
      this.gridElement.appendChild(cell);
      this.cells.push(cell);
    }

    this.container.innerHTML = "";
    const title = document.createElement("div");
    title.className = "grid-title";
    title.textContent = "DRAW A NUMBER";
    this.interactionRow = document.createElement("div");
    this.interactionRow.className = "grid-interaction-row";
    this.interactionRow.appendChild(this.gridElement);
    this.container.appendChild(title);
    this.container.appendChild(this.interactionRow);

    this.gridElement.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
    this.gridElement.addEventListener("pointermove", (event) => this.handlePointerMove(event));
    window.addEventListener("pointerup", () => this.handlePointerUp());
    this.gridElement.addEventListener("contextmenu", (event) => event.preventDefault());
  }

  setChangeHandler(handler) {
    this.onChange = handler;
  }

  getBrushSettings() {
    return {
      drawRadius: this.brush.drawRadius,
      drawStrength: this.brush.drawStrength,
      eraseRadius: this.brush.eraseRadius,
      eraseStrength: this.brush.eraseStrength,
      softness: this.brush.softness,
    };
  }

  updateBrushSettings(updates = {}) {
    if (!updates || typeof updates !== "object") {
      return this.getBrushSettings();
    }
    if (Object.prototype.hasOwnProperty.call(updates, "drawRadius")) {
      const radius = Number(updates.drawRadius);
      if (Number.isFinite(radius)) {
        this.brush.drawRadius = clamp(radius, 0.2, 10);
      }
    }
    if (Object.prototype.hasOwnProperty.call(updates, "drawStrength")) {
      const strength = Number(updates.drawStrength);
      if (Number.isFinite(strength)) {
        this.brush.drawStrength = clamp(strength, 0, 1);
      }
    }
    return this.getBrushSettings();
  }

  handlePointerDown(event) {
    event.preventDefault();
    const isErase = event.button === 2 || event.buttons === 2;
    this.activeMode = isErase ? "erase" : "draw";
    this.isDrawing = true;
    this.applyPointer(event);
  }

  handlePointerMove(event) {
    if (!this.isDrawing) return;
    this.applyPointer(event);
  }

  handlePointerUp() {
    this.isDrawing = false;
  }

  applyPointer(event) {
    const element = document.elementFromPoint(event.clientX, event.clientY);
    if (!element) return;
    const cell = element.closest("[data-index]");
    if (!cell) return;
    const index = Number(cell.dataset.index);
    if (Number.isNaN(index)) return;
    this.paintCell(index, this.activeMode === "erase");
  }

  paintCell(index, erase = false) {
    const row = Math.floor(index / this.cols);
    const col = index % this.cols;
    if (row < 0 || col < 0) return;
    const changed = this.applyBrush(row, col, erase);
    if (changed) {
      this.scheduleChange();
    }
  }

  applyBrush(centerRow, centerCol, erase = false) {
    const radius = erase ? this.brush.eraseRadius : this.brush.drawRadius;
    const strength = erase ? -this.brush.eraseStrength : this.brush.drawStrength;
    const softness = clamp(this.brush.softness ?? 0.5, 0, 0.95);
    const span = Math.ceil(radius);
    let modified = false;
    for (let row = centerRow - span; row <= centerRow + span; row += 1) {
      if (row < 0 || row >= this.rows) continue;
      for (let col = centerCol - span; col <= centerCol + span; col += 1) {
        if (col < 0 || col >= this.cols) continue;
        const distance = Math.hypot(row - centerRow, col - centerCol);
        if (distance > radius) continue;
        const falloff = 1 - distance / radius;
        if (falloff <= 0) continue;
        const influence = Math.pow(falloff, 1 + softness * 2);
        const delta = strength * influence;
        if (Math.abs(delta) < 1e-3) continue;
        const cellIndex = row * this.cols + col;
        const current = this.values[cellIndex];
        const nextValue = clamp(current + delta, 0, 1);
        if (nextValue === current) continue;
        this.values[cellIndex] = nextValue;
        this.updateCellVisual(cellIndex);
        modified = true;
      }
    }
    return modified;
  }

  updateCellVisual(index) {
    const cell = this.cells[index];
    if (!cell) return;
    const value = this.values[index];
    if (value <= 0) {
      cell.style.background = "rgba(255, 255, 255, 0.05)";
      cell.classList.remove("active");
      return;
    }
    const hue = 180 - value * 70;
    const saturation = 70 + value * 25;
    const lightness = 25 + value * 40;
    cell.style.background = `hsl(${hue.toFixed(0)}, ${saturation.toFixed(0)}%, ${lightness.toFixed(0)}%)`;
    cell.classList.add("active");
  }

  scheduleChange() {
    if (this.pendingChange) return;
    this.pendingChange = true;
    requestAnimationFrame(() => {
      this.pendingChange = false;
      if (typeof this.onChange === "function") {
        this.onChange();
      }
    });
  }

  getPixels() {
    return Float32Array.from(this.values);
  }

  setPixels(pixels) {
    if (!pixels || typeof pixels.length !== "number") {
      throw new Error("Invalid pixel values for the drawing grid.");
    }
    if (pixels.length !== this.values.length) {
      throw new Error(
        `Expected ${this.values.length} pixels, but received ${pixels.length}.`,
      );
    }
    for (let i = 0; i < this.values.length; i += 1) {
      const value = clamp(Number(pixels[i]) || 0, 0, 1);
      if (this.values[i] !== value) {
        this.values[i] = value;
        this.updateCellVisual(i);
      }
    }
    if (typeof this.onChange === "function") {
      this.onChange();
    }
  }

  clear() {
    this.values.fill(0);
    for (let i = 0; i < this.cells.length; i += 1) {
      this.updateCellVisual(i);
    }
    if (typeof this.onChange === "function") {
      this.onChange();
    }
  }

  getInteractionRow() {
    return this.interactionRow;
  }

  getGridElement() {
    return this.gridElement;
  }
}

class FeedForwardModel {
  constructor(definition) {
    if (!definition.layers?.length) {
      throw new Error("Network definition must contain layers.");
    }
    this.normalization = definition.normalization ?? { mean: 0, std: 1 };
    this.architecture = Array.isArray(definition.architecture)
      ? definition.architecture.slice()
      : this.computeArchitecture(definition.layers);
    this.layers = definition.layers.map((layer, index) => this.normaliseLayer(layer, index));
  }

  computeArchitecture(layers) {
    if (!layers.length) return [];
    const architecture = [];
    const firstLayer = layers[0];
    architecture.push(firstLayer.weights[0]?.length ?? 0);
    for (const layer of layers) {
      architecture.push(layer.biases.length);
    }
    return architecture;
  }

  normaliseLayer(layer, index) {
    if (!layer || !Array.isArray(layer.weights) || layer.weights.length === 0) {
      throw new Error(`Layer ${index} is missing valid weight matrices.`);
    }
    const weights = layer.weights.map((row) => {
      if (row instanceof Float32Array) {
        return new Float32Array(row);
      }
      if (Array.isArray(row)) {
        return Float32Array.from(row);
      }
      throw new Error(`Layer ${index} contains an invalid weight row.`);
    });
    let biases;
    if (layer.biases instanceof Float32Array) {
      biases = new Float32Array(layer.biases);
    } else if (Array.isArray(layer.biases)) {
      biases = Float32Array.from(layer.biases);
    } else {
      biases = new Float32Array(weights.length > 0 ? weights[0].length : 0);
    }
    return {
      name: typeof layer.name === "string" ? layer.name : `dense_${index}`,
      activation: typeof layer.activation === "string" ? layer.activation : "relu",
      weights,
      biases,
    };
  }

  updateLayers(layerDefinitions) {
    if (!Array.isArray(layerDefinitions) || layerDefinitions.length === 0) {
      throw new Error("New layer definitions must contain at least one layer.");
    }
    this.layers = layerDefinitions.map((layer, index) => this.normaliseLayer(layer, index));
    this.architecture = this.computeArchitecture(this.layers);
  }

  propagate(pixels) {
    const { mean, std } = this.normalization;
    const input = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i += 1) {
      input[i] = (pixels[i] - mean) / std;
    }

    const activations = [input];
    const preActivations = [];
    let current = input;

    for (const layer of this.layers) {
      const outSize = layer.biases.length;
      const linear = new Float32Array(outSize);

      for (let neuron = 0; neuron < outSize; neuron += 1) {
        let sum = layer.biases[neuron];
        const weights = layer.weights[neuron];
        for (let source = 0; source < weights.length; source += 1) {
          sum += weights[source] * current[source];
        }
        linear[neuron] = sum;
      }

      preActivations.push(linear);
      let activated;
      if (layer.activation === "relu") {
        activated = new Float32Array(outSize);
        for (let i = 0; i < outSize; i += 1) {
          activated[i] = linear[i] > 0 ? linear[i] : 0;
        }
      } else {
        activated = linear.slice();
      }
      activations.push(activated);
      current = activated;
    }

    return {
      normalizedInput: activations[0],
      activations,
      preActivations,
    };
  }
}

class ProbabilityPanel {
  constructor(container) {
    this.container = container;
    this.rows = [];
    if (!this.container) {
      throw new Error("Prediction chart container not found.");
    }
    this.build();
  }

  build() {
    this.container.innerHTML = "";
    const title = document.createElement("h3");
    title.textContent = "PROBABILITIES OF THE DIGITS";
    this.container.appendChild(title);

    this.chartElement = document.createElement("div");
    this.chartElement.className = "prediction-chart";
    this.container.appendChild(this.chartElement);

    for (let digit = 0; digit < 10; digit += 1) {
      const row = document.createElement("div");
      row.className = "prediction-bar-container";

      const label = document.createElement("span");
      label.className = "prediction-label";
      label.textContent = String(digit);

      const track = document.createElement("div");
      track.className = "prediction-bar-track";

      const bar = document.createElement("div");
      bar.className = "prediction-bar";
      track.appendChild(bar);

      const value = document.createElement("span");
      value.className = "prediction-percentage";
      value.textContent = "0.0%";

      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(value);
      this.chartElement.appendChild(row);
      this.rows.push({ bar, value });
    }
  }

  update(probabilities) {
    if (!probabilities.length) return;
    const maxProb = Math.max(...probabilities);
    probabilities.forEach((prob, index) => {
      const clamped = Math.max(0, Math.min(1, prob));
      const entry = this.rows[index];
      if (!entry) return;
      entry.bar.style.width = `${(clamped * 100).toFixed(1)}%`;
      entry.value.textContent = `${(clamped * 100).toFixed(1)}%`;
      if (clamped === maxProb) {
        entry.bar.classList.add("highest");
      } else {
        entry.bar.classList.remove("highest");
      }
    });
  }
}

class NetworkInfoPanel {
  constructor(container) {
    this.container = container;
    if (!this.container) {
      throw new Error("Network info container not found.");
    }
    this.numberFormatter = new Intl.NumberFormat("de-DE");
    this.build();
  }

  build() {
    this.container.innerHTML = "";
    if (!this.container.classList.contains("network-info-panel")) {
      this.container.classList.add("network-info-panel");
    }
    this.titleElement = document.createElement("h3");
    this.titleElement.className = "network-info-panel__title";
    this.titleElement.textContent = "NETWORK OVERVIEW";

    this.summaryElement = document.createElement("div");
    this.summaryElement.className = "network-info-panel__summary";

    this.layersElement = document.createElement("div");
    this.layersElement.className = "network-info-panel__layers";

    this.emptyElement = document.createElement("div");
    this.emptyElement.className = "network-info-panel__empty";
    this.emptyElement.textContent = "No network data available.";

    this.container.appendChild(this.titleElement);
    this.container.appendChild(this.summaryElement);
    this.container.appendChild(this.layersElement);
    this.container.appendChild(this.emptyElement);

    this.summaryElement.style.display = "none";
    this.layersElement.style.display = "none";
    this.emptyElement.style.display = "block";
  }

  formatNumber(value) {
    if (!Number.isFinite(value)) return "—";
    return this.numberFormatter.format(Math.round(value));
  }

  buildMetric(label, value) {
    const wrapper = document.createElement("span");
    wrapper.className = "network-info-panel__metric";
    const labelElement = document.createElement("span");
    labelElement.className = "network-info-panel__metric-label";
    labelElement.textContent = `${label}:`;
    const valueElement = document.createElement("span");
    valueElement.textContent = this.formatNumber(value);
    wrapper.appendChild(labelElement);
    wrapper.appendChild(valueElement);
    return wrapper;
  }

  buildSummaryLine(label, value) {
    const line = document.createElement("div");
    line.className = "network-info-panel__summary-line";
    const labelElement = document.createElement("strong");
    labelElement.textContent = label;
    const valueElement = document.createElement("span");
    valueElement.textContent = this.formatNumber(value);
    line.appendChild(labelElement);
    line.appendChild(valueElement);
    return line;
  }

  describeLayerName(rawName, index) {
    if (typeof rawName === "string" && rawName.trim().length > 0) {
      const normalized = rawName.replace(/[_-]+/g, " ").trim();
      return normalized.length ? normalized : `Layer ${index + 1}`;
    }
    return `Layer ${index + 1}`;
  }

  update(model) {
    if (!model || !Array.isArray(model.layers) || model.layers.length === 0) {
      this.emptyElement.style.display = "block";
      this.summaryElement.style.display = "none";
      this.layersElement.style.display = "none";
      return;
    }

    const architecture = Array.isArray(model.architecture) ? model.architecture : [];
    const layerSummaries = model.layers.map((layer, index) => {
      const archInput = architecture[index];
      const archOutput = architecture[index + 1];
      const weightRows = Array.isArray(layer.weights) ? layer.weights : [];
      const inputSize =
        typeof archInput === "number" && Number.isFinite(archInput)
          ? archInput
          : weightRows[0]?.length ?? 0;
      const outputSizeCandidate =
        typeof archOutput === "number" && Number.isFinite(archOutput)
          ? archOutput
          : layer.biases?.length ?? 0;
      const outputSize = Number.isFinite(outputSizeCandidate) ? outputSizeCandidate : 0;

      let weightCount = 0;
      for (const row of weightRows) {
        if (row && typeof row.length === "number") {
          weightCount += row.length;
        }
      }

      const biasArray = layer.biases;
      const biasCount = typeof biasArray?.length === "number" ? biasArray.length : 0;
      const parameterCount = weightCount + biasCount;
      return {
        index,
        name: this.describeLayerName(layer.name, index),
        activation: typeof layer.activation === "string" ? layer.activation : null,
        inputSize,
        outputSize,
        weightCount,
        biasCount,
        parameterCount,
      };
    });

    const totalParameters = layerSummaries.reduce((sum, entry) => sum + entry.parameterCount, 0);
    this.summaryElement.innerHTML = "";
    this.summaryElement.appendChild(this.buildSummaryLine("Total Parameters", totalParameters));
    if (architecture.length > 0) {
      const firstArchitectureValue = architecture[0];
      const lastArchitectureValue = architecture[architecture.length - 1];
      const inputNodes =
        typeof firstArchitectureValue === "number" && Number.isFinite(firstArchitectureValue)
          ? firstArchitectureValue
          : layerSummaries[0]?.inputSize ?? 0;
      const lastLayer = layerSummaries[layerSummaries.length - 1];
      const outputNodes =
        typeof lastArchitectureValue === "number" && Number.isFinite(lastArchitectureValue)
          ? lastArchitectureValue
          : lastLayer?.outputSize ?? 0;
      this.summaryElement.appendChild(this.buildSummaryLine("Input Nodes", inputNodes));
      this.summaryElement.appendChild(this.buildSummaryLine("Output Classes", outputNodes));
    }
    this.summaryElement.appendChild(this.buildSummaryLine("Layers (incl. Output)", layerSummaries.length));

    this.layersElement.innerHTML = "";
    layerSummaries.forEach((entry) => {
      const layerRow = document.createElement("div");
      layerRow.className = "network-info-panel__layer";

      const title = document.createElement("div");
      title.className = "network-info-panel__layer-title";
      const activationLabel = entry.activation ? ` (${entry.activation})` : "";
      title.textContent = `${entry.name}${activationLabel} • ${this.formatNumber(entry.inputSize)} → ${this.formatNumber(entry.outputSize)}`;

      const metrics = document.createElement("div");
      metrics.className = "network-info-panel__layer-metrics";
      metrics.appendChild(this.buildMetric("Weights", entry.weightCount));
      metrics.appendChild(this.buildMetric("Bias", entry.biasCount));
      metrics.appendChild(this.buildMetric("Total", entry.parameterCount));

      layerRow.appendChild(title);
      layerRow.appendChild(metrics);
      this.layersElement.appendChild(layerRow);
    });

    this.emptyElement.style.display = "none";
    this.summaryElement.style.display = "";
    this.layersElement.style.display = "";
  }
}

class NeuronDetailPanel {
  constructor(root) {
    this.root = root;
    this.onClear = null;
    this.handleClose = this.handleClose.bind(this);
    if (this.root) {
      this.root.classList.remove("visible");
      this.root.innerHTML = "";
    }
  }

  setOnClear(handler) {
    this.onClear = typeof handler === "function" ? handler : null;
  }

  handleClose(event) {
    event.preventDefault();
    if (typeof this.onClear === "function") {
      this.onClear();
    } else {
      this.hide();
    }
  }

  hide() {
    if (!this.root) return;
    this.root.innerHTML = "";
    this.root.classList.remove("visible");
  }

  update(payload) {
    if (!this.root) return;
    if (!payload) {
      this.hide();
      return;
    }

    const incomingRows = Array.isArray(payload.incoming)
      ? payload.incoming
          .map(
            (entry) => `
      <div class="neuron-detail-panel__row">
        <div><small>Source</small><br><strong>#${entry.sourceIndex + 1}</strong></div>
        <div><small>Input</small><br>${this.formatValue(entry.sourceActivation)}</div>
        <div><small>Weight</small><br>${this.formatValue(entry.weight)}</div>
        <div><small>Product</small><br><strong>${this.formatValue(entry.contribution)}</strong></div>
      </div>
    `,
          )
          .join("")
      : "";

    const outgoingRows = Array.isArray(payload.outgoing)
      ? payload.outgoing
          .map(
            (entry) => `
      <div class="neuron-detail-panel__row">
        <div><small>Target</small><br><strong>#${entry.targetIndex + 1}</strong></div>
        <div><small>Activation (Target)</small><br>${this.formatValue(entry.targetActivation)}</div>
        <div><small>Weight</small><br>${this.formatValue(entry.weight)}</div>
        <div><small>Contribution</small><br><strong>${this.formatValue(entry.contribution)}</strong></div>
      </div>
    `,
          )
          .join("")
      : "";

    const hasIncoming = Boolean(payload.incoming?.length);
    const hasOutgoing = Boolean(payload.outgoing?.length);
    const biasMarkup =
      payload.bias !== null && payload.bias !== undefined
        ? `
      <div class="neuron-detail-panel__row neuron-detail-panel__row--bias">
        <div><small>Bias</small><br><strong>${this.formatValue(payload.bias)}</strong></div>
        <div></div>
        <div></div>
        <div></div>
      </div>
    `
        : "";
    const totalMarkup =
      payload.preActivation !== null && payload.preActivation !== undefined
        ? `
      <div class="neuron-detail-panel__row neuron-detail-panel__row--total">
        <div><small>Σ</small><br><strong>${this.formatValue(payload.preActivation)}</strong></div>
        <div></div>
        <div></div>
        <div></div>
      </div>
    `
        : "";

    const totalsBlock =
      biasMarkup || totalMarkup
        ? `
      <div class="neuron-detail-panel__totals">
        ${biasMarkup}
        ${totalMarkup}
      </div>
    `
        : "";

    const incomingSection = hasIncoming
      ? `
      <div>
        <div class="neuron-detail-panel__section-title">Incoming Contributions</div>
        <div class="neuron-detail-panel__row neuron-detail-panel__row--header">
          <div>Source</div>
          <div>Input</div>
          <div>Weight</div>
          <div>Product</div>
        </div>
        ${incomingRows}
      </div>
    `
      : `<div class="neuron-detail-panel__empty">No incoming connections for this layer.</div>`;

    const outgoingSection = hasOutgoing
      ? `
      <div>
        <div class="neuron-detail-panel__section-title">Outgoing Contributions</div>
        <div class="neuron-detail-panel__row neuron-detail-panel__row--header">
          <div>Target</div>
          <div>Activation (Target)</div>
          <div>Weight</div>
          <div>Contribution</div>
        </div>
        ${outgoingRows}
      </div>
    `
      : "";

    const summaryFormula =
      payload.preActivation !== null && payload.preActivation !== undefined
        ? `Σ = Σ(input × weight)${payload.bias !== null && payload.bias !== undefined ? " + bias" : ""}`
        : "";

    this.root.innerHTML = `
      <div class="neuron-detail-panel__inner">
        <div class="neuron-detail-panel__header">
          <div class="neuron-detail-panel__title">${payload.layerLabel} • Neuron ${payload.neuronIndex + 1}${
            payload.activationName ? ` (${payload.activationName})` : ""
          }</div>
          <button type="button" class="neuron-detail-panel__close">Deselect</button>
        </div>
        <div class="neuron-detail-panel__body">
          <div class="neuron-detail-panel__summary">
            ${summaryFormula ? `<div>${summaryFormula}</div>` : ""}
          </div>
          ${totalsBlock}
          <div class="neuron-detail-panel__activations">
            <span>Input Layer Size: ${payload.previousLayerSize ?? "—"}</span>
            <span>Output Layer Size: ${payload.nextLayerSize ?? "—"}</span>
          </div>
          ${incomingSection}
          ${outgoingSection}
        </div>
      </div>
    `;

    this.root.classList.add("visible");
    const closeButton = this.root.querySelector(".neuron-detail-panel__close");
    if (closeButton) {
      closeButton.addEventListener("click", this.handleClose);
    }
  }

  formatValue(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    if (!Number.isFinite(value)) return value > 0 ? "∞" : value < 0 ? "-∞" : "NaN";
    const abs = Math.abs(value);
    if (abs >= 10000 || (abs > 0 && abs < 0.0001)) {
      return value.toExponential(2);
    }
    return value.toFixed(abs >= 1 ? 3 : 4);
  }
}

class FpsMonitor {
  constructor() {
    this.frameCount = 0;
    this.accumulatedTime = 0;
    this.lastTimestamp = 0;
    this.lastFrameTimestamp = 0;
    this.currentFps = null;

    this.root = document.createElement("div");
    this.root.className = "fps-overlay";

    this.valueElement = document.createElement("span");
    this.valueElement.className = "fps-overlay__value";
    this.valueElement.textContent = "— fps";

    this.root.appendChild(this.valueElement);
    document.body.appendChild(this.root);
    this.refreshDisplay = this.refreshDisplay.bind(this);
    this.displayTimer = window.setInterval(this.refreshDisplay, 250);
    this.refreshDisplay();
  }

  update(time) {
    if (!Number.isFinite(time)) return;
    if (this.lastTimestamp === 0) {
      this.lastTimestamp = time;
      this.lastFrameTimestamp = time;
      return;
    }
    const delta = time - this.lastTimestamp;
    this.lastTimestamp = time;
    if (delta < 0) return;

    this.accumulatedTime += delta;
    this.frameCount += 1;
    this.lastFrameTimestamp = time;

    if (this.accumulatedTime >= 250) {
      const fps = Math.round((this.frameCount * 1000) / this.accumulatedTime);
      this.currentFps = fps;
      this.accumulatedTime = 0;
      this.frameCount = 0;
      this.refreshDisplay();
    }
  }

  refreshDisplay() {
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const timeSinceLastFrame =
      this.lastFrameTimestamp > 0 ? now - this.lastFrameTimestamp : Number.POSITIVE_INFINITY;

    if (!Number.isFinite(timeSinceLastFrame) || timeSinceLastFrame > 600) {
      this.valueElement.textContent = "idle";
      this.currentFps = null;
      return;
    }

    if (this.currentFps !== null) {
      this.valueElement.textContent = `${this.currentFps} fps`;
    } else {
      this.valueElement.textContent = "— fps";
    }
  }
}

class NeuralVisualizer {
  constructor(mlp, options) {
    this.mlp = mlp;
    this.options = Object.assign(
      {
        layerSpacing: 5.5,
        inputSpacing: 0.24,
        hiddenSpacing: 0.95,
        outputSpacing: 0.95,
        inputNodeSize: 0.18,
        hiddenNodeRadius: 0.22,
        maxConnectionsPerNeuron: 24,
        connectionRadius: 0.005,
        connectionWeightThreshold: 0,
        outputLabelOffset: 0.65,
        outputLabelScale: 0.48,
        showFpsOverlay: false,
      },
      options || {},
    );
    this.focusChangeCallback =
      typeof this.options.onNeuronFocusChange === "function" ? this.options.onNeuronFocusChange : null;
    delete this.options.onNeuronFocusChange;
    this.layerMeshes = [];
    this.connectionGroups = [];
    this.selectionConnectionGroups = [];
    this.tempObject = new THREE.Object3D();
    this.tempColor = new THREE.Color();
    this.tempQuaternion = new THREE.Quaternion();
    this.upVector = new THREE.Vector3(0, 1, 0);
    this.highlightColor = new THREE.Color(0x4da6ff);
    this.outputLabels = [];
    this.selectedNeuron = null;
    this.lastDisplayActivations = null;
    this.lastNetworkActivations = null;
    this.lastPreActivations = null;
    this.currentSelectionDetail = null;
    this.selectionCylinderGeometry = null;
    this.selectionConnectionRadiusMultiplier = 1.2;
    this.selectionConnectionData = null;
    this.selectionGlowSprite = null;
    this.maxConnectionWeightMagnitude = 0;
    this.raycaster = new THREE.Raycaster();
    this.pointerVector = new THREE.Vector2();
    this.pointerDown = null;
    this.initThreeScene();
    this.buildLayers();
    this.buildConnections();
    this.animate();
  }

  initThreeScene() {
    this.scene = new THREE.Scene();
    this.scene.background = null;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this.renderer.domElement);
    this.fpsMonitor = this.options.showFpsOverlay ? new FpsMonitor() : null;

    this.labelGroup = new THREE.Group();
    this.scene.add(this.labelGroup);

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);
    this.camera.position.set(-15, 0, 15);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 52;
    this.controls.target.set(0, 0, 0);

    const ambient = new THREE.AmbientLight(0xffffff, 1.2);
    this.scene.add(ambient);
    const hemisphere = new THREE.HemisphereLight(0xffffff, 0x1a1d2e, 0.9);
    hemisphere.position.set(0, 20, 0);
    this.scene.add(hemisphere);
    const directional = new THREE.DirectionalLight(0xffffff, 1.4);
    directional.position.set(18, 26, 24);
    directional.castShadow = true;
    this.scene.add(directional);
    const fillLight = new THREE.DirectionalLight(0xa8c5ff, 0.8);
    fillLight.position.set(-20, 18, -18);
    this.scene.add(fillLight);
    const rimLight = new THREE.PointLight(0x88a4ff, 0.6, 60, 1.6);
    rimLight.position.set(0, 12, -24);
    this.scene.add(rimLight);

    this.renderer.domElement.addEventListener("pointerdown", (event) => this.handleScenePointerDown(event));
    this.renderer.domElement.addEventListener("pointerup", (event) => this.handleScenePointerUp(event));
    window.addEventListener("resize", () => this.handleResize());
    window.addEventListener("keydown", (event) => {
      if (event.defaultPrevented) return;
      if (event.key === "Escape") {
        this.clearSelection();
      }
    });
  }

  handleScenePointerDown(event) {
    if (!event.isPrimary && event.isPrimary !== undefined) return;
    if (event.button !== 0) return;
    this.pointerDown = { x: event.clientX, y: event.clientY };
  }

  handleScenePointerUp(event) {
    if (!event.isPrimary && event.isPrimary !== undefined) return;
    if (event.button !== 0) return;
    if (!this.pointerDown) return;
    const dx = event.clientX - this.pointerDown.x;
    const dy = event.clientY - this.pointerDown.y;
    this.pointerDown = null;
    const distance = Math.hypot(dx, dy);
    if (distance > 4) return;
    this.trySelectNeuron(event);
  }

  trySelectNeuron(event) {
    if (!this.layerMeshes.length || !this.camera) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.pointerVector.set(x, y);
    this.raycaster.setFromCamera(this.pointerVector, this.camera);

    const intersections = [];
    this.layerMeshes.forEach((layer, layerIndex) => {
      if (!layer?.mesh) return;
      const hits = this.raycaster.intersectObject(layer.mesh, false);
      hits.forEach((hit) => {
        if (Number.isInteger(hit.instanceId)) {
          intersections.push({
            distance: hit.distance,
            layerIndex,
            neuronIndex: hit.instanceId,
          });
        }
      });
    });
    intersections.sort((a, b) => a.distance - b.distance);
    const hit = intersections[0];
    if (!hit) {
      this.clearSelection();
      return;
    }
    this.setSelectedNeuron(hit.layerIndex, hit.neuronIndex);
  }

  setSelectedNeuron(layerIndex, neuronIndex) {
    if (!Number.isInteger(layerIndex) || !Number.isInteger(neuronIndex)) return;
    const layer = this.layerMeshes[layerIndex];
    if (!layer) return;
    const boundedIndex = Math.max(0, Math.min(layer.positions.length - 1, neuronIndex));
    if (
      this.selectedNeuron &&
      this.selectedNeuron.layerIndex === layerIndex &&
      this.selectedNeuron.neuronIndex === boundedIndex
    ) {
      this.clearSelection();
      return;
    }
    this.selectedNeuron = { layerIndex, neuronIndex: boundedIndex };
    this.updateConnectionVisibility();
    this.buildSelectionConnectionMeshes();
    if (this.lastDisplayActivations && this.lastNetworkActivations) {
      this.update(this.lastDisplayActivations, this.lastNetworkActivations, this.lastPreActivations);
    } else if (typeof this.requestRender === "function") {
      this.requestRender();
    }
  }

  clearSelection() {
    if (!this.selectedNeuron) return;
    this.selectedNeuron = null;
    this.selectionConnectionData = null;
    this.currentSelectionDetail = null;
    this.disposeSelectionConnectionMeshes();
    this.updateConnectionVisibility();
    this.hideSelectionGlow();
    if (typeof this.focusChangeCallback === "function") {
      this.focusChangeCallback(null);
    }
    if (this.lastDisplayActivations && this.lastNetworkActivations) {
      this.update(this.lastDisplayActivations, this.lastNetworkActivations, this.lastPreActivations);
    } else if (typeof this.requestRender === "function") {
      this.requestRender();
    }
  }

  updateConnectionVisibility() {
    const showDefaultConnections = !this.selectedNeuron;
    this.connectionGroups.forEach((group) => {
      if (!group?.mesh) return;
      group.mesh.visible = showDefaultConnections;
    });
  }

  ensureSelectionGeometry() {
    if (!this.selectionCylinderGeometry) {
      const baseRadius = this.options.connectionRadius ?? 0.02;
      const radius = baseRadius * this.selectionConnectionRadiusMultiplier;
      this.selectionCylinderGeometry = new THREE.CylinderGeometry(radius, radius, 1, 16, 1, true);
    }
    return this.selectionCylinderGeometry;
  }

  buildSelectionConnectionMeshes() {
    this.disposeSelectionConnectionMeshes();
    if (!this.selectedNeuron) return;
    const data = this.collectSelectionConnectionData();
    this.selectionConnectionData = data;
    const baseGeometry = this.ensureSelectionGeometry();
    const connectionMaterial = new THREE.MeshBasicMaterial({
      toneMapped: false,
      vertexColors: true,
    });
    connectionMaterial.vertexColors = true;

    const createMesh = (connections) => {
      if (!connections.length) return null;
      const mesh = new THREE.InstancedMesh(baseGeometry.clone(), connectionMaterial.clone(), connections.length);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(connections.length * 3), 3);
      colorAttribute.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = colorAttribute;
      connections.forEach((connection, index) => {
        const direction = connection.targetPosition.clone().sub(connection.sourcePosition);
        const length = direction.length();
        if (length <= 0) {
          return;
        }
        const midpoint = connection.sourcePosition.clone().addScaledVector(direction, 0.5);
        this.tempObject.position.copy(midpoint);
        this.tempQuaternion.setFromUnitVectors(this.upVector, direction.clone().normalize());
        this.tempObject.quaternion.copy(this.tempQuaternion);
        this.tempObject.scale.set(1, length, 1);
        this.tempObject.updateMatrix();
        mesh.setMatrixAt(index, this.tempObject.matrix);
        mesh.setColorAt(index, this.tempColor.setRGB(0.8, 0.8, 0.8));
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
      return mesh;
    };

    if (data.incoming.length) {
      const mesh = createMesh(data.incoming);
      if (mesh) {
        this.scene.add(mesh);
        this.selectionConnectionGroups.push({
          mesh,
          connections: data.incoming,
          direction: "incoming",
        });
      }
    }
    if (data.outgoing.length) {
      const mesh = createMesh(data.outgoing);
      if (mesh) {
        this.scene.add(mesh);
        this.selectionConnectionGroups.push({
          mesh,
          connections: data.outgoing,
          direction: "outgoing",
        });
      }
    }
    this.updateSelectionGlow();
  }

  disposeSelectionConnectionMeshes() {
    if (!Array.isArray(this.selectionConnectionGroups)) {
      this.selectionConnectionGroups = [];
      return;
    }
    this.selectionConnectionGroups.forEach((group) => {
      if (!group?.mesh) return;
      this.scene.remove(group.mesh);
      const material = group.mesh.material;
      if (Array.isArray(material)) {
        material.forEach((mat) => {
          if (mat && typeof mat.dispose === "function") mat.dispose();
        });
      } else if (material && typeof material.dispose === "function") {
        material.dispose();
      }
      if (group.mesh.geometry && typeof group.mesh.geometry.dispose === "function") {
        group.mesh.geometry.dispose();
      }
    });
    this.selectionConnectionGroups = [];
  }

  collectSelectionConnectionData() {
    if (!this.selectedNeuron) {
      return { incoming: [], outgoing: [], bias: 0, previousLayerSize: null, nextLayerSize: null };
    }
    const { layerIndex, neuronIndex } = this.selectedNeuron;
    const targetLayerMesh = this.layerMeshes[layerIndex];
    const targetPosition = targetLayerMesh?.positions?.[neuronIndex];
    const incoming = [];
    const outgoing = [];
    const minMagnitude = Math.max(0, this.options.connectionWeightThreshold ?? 0);
    let bias = null;
    let previousLayerSize = null;
    let nextLayerSize = null;

    if (layerIndex > 0 && this.mlp.layers[layerIndex - 1]) {
      const prevLayerMesh = this.layerMeshes[layerIndex - 1];
      const weightLayer = this.mlp.layers[layerIndex - 1];
      const weights = weightLayer.weights?.[neuronIndex];
      bias = weightLayer.biases?.[neuronIndex] ?? 0;
      if (weights && targetPosition) {
        previousLayerSize = weights.length;
        for (let sourceIndex = 0; sourceIndex < weights.length; sourceIndex += 1) {
          const sourcePosition = prevLayerMesh?.positions?.[sourceIndex];
          if (!sourcePosition) continue;
          const weight = Number(weights[sourceIndex]);
          if (!Number.isFinite(weight)) continue;
          if (Math.abs(weight) < minMagnitude) continue;
          incoming.push({
            sourceLayer: layerIndex - 1,
            targetLayer: layerIndex,
            sourceIndex,
            targetIndex: neuronIndex,
            weight,
            sourcePosition,
            targetPosition,
          });
        }
      }
    }

    if (layerIndex < this.layerMeshes.length - 1 && this.mlp.layers[layerIndex]) {
      const nextLayerMesh = this.layerMeshes[layerIndex + 1];
      const weightLayer = this.mlp.layers[layerIndex];
      const weights = weightLayer.weights ?? [];
      nextLayerSize = weights.length;
      const sourcePosition = targetPosition;
      for (let targetIndex = 0; targetIndex < weights.length; targetIndex += 1) {
        const row = weights[targetIndex];
        if (!row || sourcePosition == null) continue;
        const targetPosition = nextLayerMesh?.positions?.[targetIndex];
        if (!targetPosition) continue;
        const weight = Number(row[neuronIndex]);
        if (!Number.isFinite(weight)) continue;
        if (Math.abs(weight) < minMagnitude) continue;
        outgoing.push({
          sourceLayer: layerIndex,
          targetLayer: layerIndex + 1,
          sourceIndex: neuronIndex,
          targetIndex,
          weight,
          sourcePosition,
          targetPosition,
        });
      }
    }

    return { incoming, outgoing, bias, previousLayerSize, nextLayerSize };
  }

  ensureSelectionGlowSprite() {
    if (this.selectionGlowSprite) return this.selectionGlowSprite;
    const size = 256;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, size, size);
      const gradient = ctx.createRadialGradient(size / 2, size / 2, size * 0.12, size / 2, size / 2, size * 0.5);
      gradient.addColorStop(0, "rgba(77, 166, 255, 0.95)");
      gradient.addColorStop(0.35, "rgba(39, 132, 255, 0.7)");
      gradient.addColorStop(1, "rgba(18, 64, 158, 0)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, size, size);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    texture.anisotropy = 2;
    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      opacity: 0.95,
      toneMapped: false,
      color: 0x4da6ff,
    });
    const sprite = new THREE.Sprite(material);
    sprite.visible = false;
    sprite.renderOrder = 10;
    this.scene.add(sprite);
    this.selectionGlowSprite = sprite;
    return sprite;
  }

  updateSelectionGlow() {
    if (!this.selectedNeuron) {
      this.hideSelectionGlow();
      return;
    }
    const { layerIndex, neuronIndex } = this.selectedNeuron;
    const layer = this.layerMeshes[layerIndex];
    const position = layer?.positions?.[neuronIndex];
    if (!layer || !position) {
      this.hideSelectionGlow();
      return;
    }
    const sprite = this.ensureSelectionGlowSprite();
    sprite.position.copy(position);
    const baseSize =
      layer.type === "input"
        ? this.options.inputNodeSize ?? 0.18
        : this.options.hiddenNodeRadius ?? 0.22;
    const scale = Math.max(baseSize * 3, 0.2);
    sprite.scale.set(scale, scale, 1);
    sprite.visible = true;
  }

  hideSelectionGlow() {
    if (!this.selectionGlowSprite) return;
    this.selectionGlowSprite.visible = false;
  }

  updateSelectionVisuals() {
    if (!this.selectedNeuron) return;
    if (!this.lastNetworkActivations) return;
    if (!this.selectionConnectionData) {
      this.selectionConnectionData = this.collectSelectionConnectionData();
    }
    this.updateSelectionGlow();
    const detail = this.buildSelectionDetail();
    this.currentSelectionDetail = detail;
    this.updateSelectedConnectionColors(detail);
    if (typeof this.focusChangeCallback === "function") {
      this.focusChangeCallback(detail);
    }
    if (typeof this.requestRender === "function") {
      this.requestRender();
    }
  }

  buildSelectionDetail() {
    if (!this.selectedNeuron) return null;
    const { layerIndex, neuronIndex } = this.selectedNeuron;
    const data = this.selectionConnectionData || this.collectSelectionConnectionData();
    const layerActivations = this.lastNetworkActivations?.[layerIndex] ?? null;
    const activationValue =
      Array.isArray(layerActivations) || layerActivations instanceof Float32Array
        ? layerActivations[neuronIndex] ?? null
        : null;
    const previousActivations =
      layerIndex > 0 ? this.lastNetworkActivations?.[layerIndex - 1] ?? null : null;
    const nextActivations =
      layerIndex < this.lastNetworkActivations.length - 1
        ? this.lastNetworkActivations?.[layerIndex + 1] ?? null
        : null;
    let sumContributions = 0;

    const incoming = data.incoming.map((connection) => {
      const sourceActivation =
        previousActivations && (previousActivations instanceof Float32Array || Array.isArray(previousActivations))
          ? previousActivations[connection.sourceIndex] ?? 0
          : 0;
      const contribution = sourceActivation * connection.weight;
      sumContributions += contribution;
      return {
        sourceIndex: connection.sourceIndex,
        weight: connection.weight,
        sourceActivation,
        contribution,
      };
    });

    const bias = data.bias ?? (layerIndex > 0 ? 0 : null);
    const preActivationFromModel =
      layerIndex > 0
        ? this.lastPreActivations?.[layerIndex - 1]?.[neuronIndex] ?? null
        : activationValue ?? null;
    const inferredPreActivation =
      bias !== null && bias !== undefined ? sumContributions + bias : sumContributions;
    const preActivation =
      preActivationFromModel !== null && preActivationFromModel !== undefined
        ? preActivationFromModel
        : inferredPreActivation;

    const outgoingContributionBase = Number.isFinite(activationValue) ? activationValue : 0;
    const outgoing = data.outgoing.map((connection) => {
      const targetActivation =
        nextActivations && (nextActivations instanceof Float32Array || Array.isArray(nextActivations))
          ? nextActivations[connection.targetIndex] ?? 0
          : 0;
      return {
        targetIndex: connection.targetIndex,
        weight: connection.weight,
        targetActivation,
        contribution: outgoingContributionBase * connection.weight,
      };
    });

    return {
      layerIndex,
      neuronIndex,
      layerLabel: this.describeLayer(layerIndex),
      activationName: this.getActivationName(layerIndex),
      activationValue: activationValue ?? null,
      preActivation,
      bias: bias ?? null,
      incoming,
      outgoing,
      previousLayerSize: data.previousLayerSize,
      nextLayerSize: data.nextLayerSize,
    };
  }

  updateSelectedConnectionColors(detail) {
    if (!detail) return;
    const incomingGroup = this.selectionConnectionGroups.find((group) => group.direction === "incoming");
    if (incomingGroup && detail.incoming.length === incomingGroup.connections.length) {
      const maxContribution = detail.incoming.reduce(
        (acc, item) => Math.max(acc, Math.abs(item.contribution)),
        0,
      );
      const scale = maxContribution > 1e-6 ? maxContribution : 1;
      detail.incoming.forEach((item, index) => {
        const normalized = clamp(item.contribution / scale, -1, 1);
        const magnitude = Math.abs(normalized);
        if (magnitude < 1e-4) {
          this.tempColor.setRGB(0.4, 0.4, 0.4);
        } else if (normalized >= 0) {
          this.tempColor.setRGB(0.2, 0.85 * magnitude + 0.15, 0.2);
        } else {
          this.tempColor.setRGB(0.85 * magnitude + 0.15, 0.2, 0.2);
        }
        incomingGroup.mesh.setColorAt(index, this.tempColor);
      });
      incomingGroup.mesh.instanceColor.needsUpdate = true;
    }

    const outgoingGroup = this.selectionConnectionGroups.find((group) => group.direction === "outgoing");
    if (outgoingGroup && detail.outgoing.length === outgoingGroup.connections.length) {
      const maxContribution = detail.outgoing.reduce(
        (acc, item) => Math.max(acc, Math.abs(item.contribution)),
        0,
      );
      const scale = maxContribution > 1e-6 ? maxContribution : 1;
      detail.outgoing.forEach((item, index) => {
        const normalized = clamp(item.contribution / scale, -1, 1);
        const magnitude = Math.abs(normalized);
        if (magnitude < 1e-4) {
          this.tempColor.setRGB(0.35, 0.35, 0.35);
        } else if (normalized >= 0) {
          this.tempColor.setRGB(0.25, 0.75 * magnitude + 0.25, 0.9);
        } else {
          this.tempColor.setRGB(0.9, 0.3, 0.85 * magnitude + 0.15);
        }
        outgoingGroup.mesh.setColorAt(index, this.tempColor);
      });
      outgoingGroup.mesh.instanceColor.needsUpdate = true;
    }
  }

  describeLayer(layerIndex) {
    if (layerIndex === 0) {
      return `Eingabeschicht (${this.mlp.architecture[layerIndex]} Knoten)`;
    }
    if (layerIndex === this.mlp.architecture.length - 1) {
      return `Ausgabeschicht (${this.mlp.architecture[layerIndex]} Knoten)`;
    }
    return `Verborgene Schicht ${layerIndex} (${this.mlp.architecture[layerIndex]} Knoten)`;
  }

  getActivationName(layerIndex) {
    if (layerIndex === 0) return null;
    const activation = this.mlp.layers?.[layerIndex - 1]?.activation;
    if (typeof activation !== "string") return null;
    return activation.toUpperCase();
  }

  buildLayers() {
    const inputGeometry = new THREE.BoxGeometry(
      this.options.inputNodeSize,
      this.options.inputNodeSize,
      this.options.inputNodeSize,
    );
    const hiddenGeometry = new THREE.SphereGeometry(this.options.hiddenNodeRadius, 16, 16);
    // Test with MeshBasicMaterial for hidden/output neurons (no lighting influence)
    // Enable vertexColors so per-instance colors are applied.
    const hiddenBaseMaterial = new THREE.MeshBasicMaterial({ vertexColors: true });
    hiddenBaseMaterial.vertexColors = true;
    hiddenBaseMaterial.toneMapped = false;

    const layerCount = this.mlp.architecture.length;
    const totalWidth = (layerCount - 1) * this.options.layerSpacing;
    const startX = -totalWidth / 2;

    this.clearOutputLabels();
    this.mlp.architecture.forEach((neuronCount, layerIndex) => {
      const layerX = startX + layerIndex * this.options.layerSpacing;
      const positions = this.computeLayerPositions(layerIndex, neuronCount, layerX);
      const isOutputLayer = layerIndex === layerCount - 1;

      if (layerIndex === 0) {
        // Enable vertex colors so per-instance colors are visible on neurons
        const material = new THREE.MeshLambertMaterial({ vertexColors: true });
        material.vertexColors = true;
        material.emissive.setRGB(0.08, 0.08, 0.08);
        const mesh = new THREE.InstancedMesh(inputGeometry, material, neuronCount);
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(neuronCount * 3), 3);
        colorAttribute.setUsage(THREE.DynamicDrawUsage);
        mesh.instanceColor = colorAttribute;

        positions.forEach((position, instanceIndex) => {
          this.tempObject.position.copy(position);
          this.tempObject.updateMatrix();
          mesh.setMatrixAt(instanceIndex, this.tempObject.matrix);
          mesh.setColorAt(instanceIndex, this.tempColor.setRGB(0.15, 0.15, 0.15));
        });

        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor.needsUpdate = true;
        this.scene.add(mesh);
        this.layerMeshes.push({ mesh, positions, type: "input", layerIndex });
      } else {
        const material = hiddenBaseMaterial.clone();
        // Clone geometry per mesh so each InstancedMesh can have its own instanceColor attribute
        const geometry = hiddenGeometry.clone();
        const mesh = new THREE.InstancedMesh(geometry, material, neuronCount);
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(neuronCount * 3), 3);
        colorAttribute.setUsage(THREE.DynamicDrawUsage);
        mesh.instanceColor = colorAttribute;

        positions.forEach((position, instanceIndex) => {
          this.tempObject.position.copy(position);
          this.tempObject.updateMatrix();
          mesh.setMatrixAt(instanceIndex, this.tempObject.matrix);
          mesh.setColorAt(instanceIndex, this.tempColor.setRGB(0.15, 0.15, 0.15));
        });

        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor.needsUpdate = true;
        this.scene.add(mesh);
        const layerType = isOutputLayer ? "output" : "hidden";
        this.layerMeshes.push({ mesh, positions, type: layerType, layerIndex });
        if (isOutputLayer) {
          this.createOutputLabels(positions);
        }
      }
    });
  }

  computeLayerPositions(layerIndex, neuronCount, layerX) {
    const positions = [];
    const isOutputLayer = layerIndex === this.mlp.architecture.length - 1;
    if (layerIndex === 0) {
      const spacing = this.options.inputSpacing;
      let rows;
      let cols;
      if (neuronCount === 28 * 28) {
        rows = 28;
        cols = 28;
      } else {
        cols = Math.ceil(Math.sqrt(neuronCount));
        rows = Math.ceil(neuronCount / cols);
      }
      const height = (rows - 1) * spacing;
      const width = (cols - 1) * spacing;
      let filled = 0;
      for (let row = 0; row < rows && filled < neuronCount; row += 1) {
        for (let col = 0; col < cols && filled < neuronCount; col += 1) {
          const y = height / 2 - row * spacing;
          const z = -width / 2 + col * spacing;
          positions.push(new THREE.Vector3(layerX, y, z));
          filled += 1;
        }
      }
    } else if (isOutputLayer) {
      const spacing = this.options.outputSpacing ?? this.options.hiddenSpacing;
      const height = (neuronCount - 1) * spacing;
      for (let index = 0; index < neuronCount; index += 1) {
        const y = height / 2 - index * spacing;
        positions.push(new THREE.Vector3(layerX, y, 0));
      }
    } else {
      const spacing = this.options.hiddenSpacing;
      const cols = Math.max(1, Math.ceil(Math.sqrt(neuronCount)));
      const rows = Math.ceil(neuronCount / cols);
      const height = (rows - 1) * spacing;
      const width = (cols - 1) * spacing;
      for (let index = 0; index < neuronCount; index += 1) {
        const row = Math.floor(index / cols);
        const col = index % cols;
        const y = height / 2 - row * spacing;
        const z = -width / 2 + col * spacing;
        positions.push(new THREE.Vector3(layerX, y, z));
      }
    }
    return positions;
  }

  clearOutputLabels() {
    if (!this.outputLabels.length || !this.labelGroup) return;
    this.outputLabels.forEach((sprite) => {
      if (sprite.material.map) {
        sprite.material.map.dispose();
      }
      sprite.material.dispose();
      this.labelGroup.remove(sprite);
    });
    this.outputLabels = [];
  }

  createOutputLabels(positions) {
    if (!this.labelGroup) return;
    const offset = this.options.outputLabelOffset ?? 0.65;
    const scale = this.options.outputLabelScale ?? 0.48;
    positions.forEach((position, index) => {
      const label = this.buildDigitSprite(String(index));
      label.position.copy(position);
      label.position.x += offset;
      label.scale.set(scale, scale, scale);
      this.labelGroup.add(label);
      this.outputLabels.push(label);
    });
  }

  buildDigitSprite(text) {
    const size = 128;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Canvas 2D context unavailable for label rendering.");
    }
    ctx.clearRect(0, 0, size, size);
    ctx.font = `900 ${Math.floor(size * 0.62)}px "Inter", "Segoe UI", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(235, 245, 255, 0.95)";
    ctx.strokeStyle = "rgba(12, 25, 44, 0.85)";
    ctx.lineWidth = size * 0.08;
    ctx.shadowColor = "rgba(0, 0, 0, 0.55)";
    ctx.shadowBlur = size * 0.12;
    ctx.strokeText(text, size / 2, size / 2);
    ctx.fillText(text, size / 2, size / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    texture.anisotropy = 2;
    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;

    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.center.set(0, 0.5);
    return sprite;
  }

  buildConnections() {
    this.maxConnectionWeightMagnitude = 0;
    const connectionRadius = this.options.connectionRadius ?? 0.005;
    const baseGeometry = new THREE.CylinderGeometry(connectionRadius, connectionRadius, 1, 10, 1, true);
    // Use a material that supports vertex colors so instanceColor is rendered
    const material = new THREE.MeshLambertMaterial({ vertexColors: true });
    material.vertexColors = true;

    this.mlp.layers.forEach((layer, layerIndex) => {
      const { selected, maxAbsWeight } = this.findImportantConnections(layer);
      if (Number.isFinite(maxAbsWeight) && maxAbsWeight > this.maxConnectionWeightMagnitude) {
        this.maxConnectionWeightMagnitude = maxAbsWeight;
      }
      if (!selected.length) return;

      // Clone geometry per mesh so instanceColor can be bound independently
      const mesh = new THREE.InstancedMesh(baseGeometry.clone(), material.clone(), selected.length);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(selected.length * 3), 3);
      colorAttribute.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = colorAttribute;

      selected.forEach((connection, instanceIndex) => {
        const sourcePosition = this.layerMeshes[layerIndex].positions[connection.sourceIndex];
        const targetPosition = this.layerMeshes[layerIndex + 1].positions[connection.targetIndex];
        const direction = targetPosition.clone().sub(sourcePosition);
        const length = direction.length();
        const midpoint = sourcePosition.clone().addScaledVector(direction, 0.5);

        this.tempObject.position.copy(midpoint);
        const quaternion = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          direction.clone().normalize(),
        );
        this.tempObject.scale.set(1, length, 1);
        this.tempObject.quaternion.copy(quaternion);
        this.tempObject.updateMatrix();
        mesh.setMatrixAt(instanceIndex, this.tempObject.matrix);
        mesh.setColorAt(instanceIndex, this.tempColor.setRGB(1, 1, 1));
      });

      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
      this.scene.add(mesh);
      this.connectionGroups.push({
        mesh,
        connections: selected,
        sourceLayer: layerIndex,
        maxAbsWeight,
      });
    });
  }

  disposeConnectionMeshes() {
    this.connectionGroups.forEach((group) => {
      this.scene.remove(group.mesh);
      if (group.mesh.geometry && typeof group.mesh.geometry.dispose === "function") {
        group.mesh.geometry.dispose();
      }
      const material = group.mesh.material;
      if (Array.isArray(material)) {
        material.forEach((mat) => {
          if (mat && typeof mat.dispose === "function") mat.dispose();
        });
      } else if (material && typeof material.dispose === "function") {
        material.dispose();
      }
    });
    this.connectionGroups = [];
  }

  updateNetworkWeights() {
    this.disposeConnectionMeshes();
    this.buildConnections();
    if (this.selectedNeuron) {
      this.updateConnectionVisibility();
      this.selectionConnectionData = null;
      this.buildSelectionConnectionMeshes();
      if (this.lastDisplayActivations && this.lastNetworkActivations) {
        this.updateSelectionVisuals();
      }
    }
  }

  setMaxConnectionsPerNeuron(limit) {
    const clamped = Math.max(1, Math.floor(limit));
    if (!Number.isFinite(clamped)) return false;
    if (clamped === this.options.maxConnectionsPerNeuron) return false;
    this.options.maxConnectionsPerNeuron = clamped;
    this.updateNetworkWeights();
    return true;
  }

  setConnectionRadius(radius) {
    if (!Number.isFinite(radius)) return false;
    const clamped = Math.max(0.0005, radius);
    if (Math.abs(clamped - this.options.connectionRadius) < 1e-6) return false;
    this.options.connectionRadius = clamped;
    if (this.selectionCylinderGeometry && typeof this.selectionCylinderGeometry.dispose === "function") {
      this.selectionCylinderGeometry.dispose();
    }
    this.selectionCylinderGeometry = null;
    this.updateNetworkWeights();
    return true;
  }

  setConnectionWeightThreshold(threshold) {
    if (!Number.isFinite(threshold)) return false;
    const clamped = Math.max(0, threshold);
    if (Math.abs(clamped - (this.options.connectionWeightThreshold ?? 0)) < 1e-6) return false;
    this.options.connectionWeightThreshold = clamped;
    this.updateNetworkWeights();
    return true;
  }

  getMaxConnectionWeightMagnitude() {
    return this.maxConnectionWeightMagnitude || 0;
  }

  findImportantConnections(layer) {
    const limit = this.options.maxConnectionsPerNeuron;
    const minMagnitude = Math.max(0, this.options.connectionWeightThreshold ?? 0);
    const selected = [];
    let maxAbsWeight = 0;
    for (let target = 0; target < layer.weights.length; target += 1) {
      const row = layer.weights[target];
      const candidates = [];
      for (let source = 0; source < row.length; source += 1) {
        const weight = row[source];
        if (!Number.isFinite(weight)) continue;
        const magnitude = Math.abs(weight);
        candidates.push({ sourceIndex: source, targetIndex: target, weight, magnitude });
        if (magnitude > maxAbsWeight) maxAbsWeight = magnitude;
      }
      candidates.sort((a, b) => b.magnitude - a.magnitude);
      const take = Math.min(limit, candidates.length);
      for (let i = 0; i < take; i += 1) {
        const candidate = candidates[i];
        if (candidate.magnitude < minMagnitude) break;
        selected.push({
          sourceIndex: candidate.sourceIndex,
          targetIndex: candidate.targetIndex,
          weight: candidate.weight,
        });
      }
    }
    return { selected, maxAbsWeight };
  }

  update(displayActivations, networkActivations = displayActivations, preActivations = null) {
    this.lastDisplayActivations = displayActivations;
    this.lastNetworkActivations = networkActivations;
    this.lastPreActivations = preActivations;
    this.layerMeshes.forEach((layer, layerIndex) => {
      const values = displayActivations[layerIndex];
      if (!values) return;
      const scale = layerIndex === 0 ? 1 : maxAbsValue(displayActivations[layerIndex]);
      this.applyNodeColors(layer, values, scale || 1, layerIndex);
    });

    this.connectionGroups.forEach((group) => {
      const sourceValues = networkActivations[group.sourceLayer];
      if (!sourceValues) return;
      this.applyConnectionColors(group, sourceValues);
    });
    if (this.selectedNeuron) {
      this.updateSelectionVisuals();
    } else if (typeof this.focusChangeCallback === "function" && this.currentSelectionDetail !== null) {
      this.currentSelectionDetail = null;
      this.focusChangeCallback(null);
    }
    if (typeof this.requestRender === "function") {
      this.requestRender();
    }
  }

  applyNodeColors(layer, values, scale, layerIndex) {
    const { mesh, type } = layer;
    const activeSelection = this.selectedNeuron;
    if (type === "input") {
      for (let i = 0; i < values.length; i += 1) {
        const value = clamp(values[i], 0, 1);
        const isSelected =
          activeSelection &&
          layerIndex === activeSelection.layerIndex &&
          i === activeSelection.neuronIndex;
        if (isSelected) {
          this.tempColor.copy(this.highlightColor);
        } else {
          this.tempColor.setRGB(value, value, value);
        }
        mesh.setColorAt(i, this.tempColor);
      }
      mesh.instanceColor.needsUpdate = true;
      return;
    }

    const safeScale = scale > 1e-6 ? scale : 1;
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      const normalized = clamp(value / safeScale, 0, 1);
      const isSelected =
        activeSelection &&
        layerIndex === activeSelection.layerIndex &&
        i === activeSelection.neuronIndex;
      if (isSelected) {
        this.tempColor.copy(this.highlightColor);
      } else {
        this.tempColor.setRGB(normalized, normalized, normalized);
      }
      mesh.setColorAt(i, this.tempColor);
    }
    mesh.instanceColor.needsUpdate = true;
  }

  applyConnectionColors(group, sourceValues) {
    const contributions = new Float32Array(group.connections.length);
    let maxContribution = 0;
    group.connections.forEach((connection, index) => {
      const activation = sourceValues[connection.sourceIndex] ?? 0;
      const contribution = activation * connection.weight;
      contributions[index] = contribution;
      const magnitude = Math.abs(contribution);
      if (magnitude > maxContribution) maxContribution = magnitude;
    });
    const scale = maxContribution > 1e-6 ? maxContribution : group.maxAbsWeight || 1;
    group.connections.forEach((connection, index) => {
      const normalized = clamp(contributions[index] / scale, -1, 1);
      const magnitude = Math.abs(normalized);
      if (magnitude < 1e-3) {
        this.tempColor.setRGB(0, 0, 0);
      } else if (normalized >= 0) {
        this.tempColor.setRGB(0, magnitude, 0);
      } else {
        this.tempColor.setRGB(magnitude, 0, 0);
      }
      group.mesh.setColorAt(index, this.tempColor);
    });
    group.mesh.instanceColor.needsUpdate = true;
  }

  handleResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (typeof this.requestRender === "function") {
      this.requestRender();
    }
  }

  animate() {
    this.renderRequested = false;
    this.needsContinuousRender = false;

    const renderFrame = (time) => {
      this.renderRequested = false;
      const controlsChanged = this.controls.update();
      this.renderer.render(this.scene, this.camera);
      if (this.fpsMonitor) {
        this.fpsMonitor.update(time);
      }
      if (this.needsContinuousRender || controlsChanged) {
        this.requestRender();
      }
    };

    this.requestRender = () => {
      if (this.renderRequested) return;
      this.renderRequested = true;
      requestAnimationFrame(renderFrame);
    };

    this.controls.addEventListener("start", () => {
      this.needsContinuousRender = true;
      this.requestRender();
    });
    this.controls.addEventListener("end", () => {
      this.needsContinuousRender = false;
      this.requestRender();
    });
    this.controls.addEventListener("change", () => {
      this.requestRender();
    });

    this.requestRender();
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// Lightweight DOM-based visualizer used as an optional fallback when WebGL
// rendering via Three.js is unavailable or problematic in the deployment.
class LegacyVisualizer {
  constructor(mlp, options = {}) {
    this.mlp = mlp;
    this.options = Object.assign(
      {
        layerSpacing: 5.5,
        inputSpacing: 0.24,
        hiddenSpacing: 0.95,
        inputNodeSize: 0.18,
        hiddenNodeRadius: 0.22,
        maxConnectionsPerNeuron: 24,
        connectionRadius: 0.005,
        connectionWeightThreshold: 0,
      },
      options || {},
    );

    this.root = document.createElement("div");
    this.root.id = "legacyVisualizer";
    this.root.className = "legacy-visualizer";
    document.body.appendChild(this.root);

    this.layerElements = [];
    this.buildLayers();
  }

  buildLayers() {
    this.root.innerHTML = "";
    const wrapper = document.createElement("div");
    wrapper.className = "legacy-visualizer__wrapper";
    this.root.appendChild(wrapper);

    const arch = Array.isArray(this.mlp.architecture) ? this.mlp.architecture : [];
    arch.forEach((count, layerIndex) => {
      const col = document.createElement("div");
      col.className = "legacy-layer";
      const neurons = [];
      for (let i = 0; i < count; i += 1) {
        const n = document.createElement("div");
        n.className = "legacy-neuron";
        col.appendChild(n);
        neurons.push(n);
      }
      wrapper.appendChild(col);
      this.layerElements.push({ col, neurons });
    });
  }

  update(displayActivations = [], networkActivations = [], preActivations = []) {
    // Color neurons based on displayActivations (0..1) for input layer
    for (let layerIndex = 0; layerIndex < this.layerElements.length; layerIndex += 1) {
      const layerEl = this.layerElements[layerIndex];
      const values = displayActivations[layerIndex] || [];
      for (let i = 0; i < layerEl.neurons.length; i += 1) {
        const nodeEl = layerEl.neurons[i];
        const v = Number(values[i]) || 0;
        const intensity = Math.max(0, Math.min(1, v));
        const light = 12 + Math.round(intensity * 60);
        const alpha = 0.08 + intensity * 0.9;
        nodeEl.style.background = `rgba(0, 245, 160, ${alpha})`;
        nodeEl.style.boxShadow = intensity > 0 ? `0 0 ${6 + intensity * 18}px rgba(0,245,160,${Math.min(0.45, alpha)})` : "none";
      }
    }
  }

  updateNetworkWeights() {
    // No-op for legacy visualizer; weights are not rendered as lines here.
  }

  setMaxConnectionsPerNeuron() {
    return false;
  }

  setConnectionRadius() {
    return false;
  }

  setConnectionWeightThreshold() {
    return false;
  }

  getMaxConnectionWeightMagnitude() {
    return 0;
  }

  clearSelection() {
    // no selection in legacy view
  }
}

function softmax(values) {
  if (!values.length) return [];
  const maxVal = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxVal));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => (sum === 0 ? 0 : value / sum));
}

function maxAbsValue(values) {
  let max = 0;
  for (let i = 0; i < values.length; i += 1) {
    const magnitude = Math.abs(values[i]);
    if (magnitude > max) {
      max = magnitude;
    }
  }
  return max;
}
