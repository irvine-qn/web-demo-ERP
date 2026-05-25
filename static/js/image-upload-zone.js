/**
 * Drag-drop, paste, browse, and camera capture for image upload dropzones.
 */
(function () {
  const ACCEPT_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  const ACCEPT_EXT = [".jpg", ".jpeg", ".png", ".webp"];

  function isImageFile(file) {
    if (!file || !file.type) {
      const name = (file?.name || "").toLowerCase();
      return ACCEPT_EXT.some((ext) => name.endsWith(ext));
    }
    return ACCEPT_TYPES.has(file.type) || file.type.startsWith("image/");
  }

  function assignToInput(fileInput, file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
  }

  function notify(config, message) {
    if (config.onError) {
      config.onError(message);
      return;
    }
    if (config.statusEl) {
      config.statusEl.textContent = message;
      return;
    }
    alert(message);
  }

  window.initImageUploadZone = function initImageUploadZone(config) {
    const {
      dropzone,
      fileInput,
      cameraInput,
      previewImg,
      onFileSelected,
      statusEl,
      onError,
    } = config;

    let previewUrl = null;
    let selectedFile = null;

    function setFile(file) {
      if (!isImageFile(file)) {
        notify({ statusEl, onError }, "Unsupported format. Use JPG, PNG, or WEBP.");
        return;
      }

      assignToInput(fileInput, file);
      if (cameraInput && cameraInput !== fileInput) {
        assignToInput(cameraInput, file);
      }

      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      previewUrl = URL.createObjectURL(file);
      previewImg.src = previewUrl;
      previewImg.style.display = "block";
      dropzone.classList.add("has-image");

      selectedFile = file;
      if (onFileSelected) {
        onFileSelected(file);
      }
    }

    function clearDragState() {
      dropzone.classList.remove("drag-over");
    }

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file) setFile(file);
    });

    if (cameraInput) {
      cameraInput.addEventListener("change", () => {
        const file = cameraInput.files[0];
        if (file) setFile(file);
      });
    }

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropzone.classList.add("drag-over");
      });
    });

    dropzone.addEventListener("dragleave", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (event.currentTarget === dropzone) {
        clearDragState();
      }
    });

    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      event.stopPropagation();
      clearDragState();
      const file = event.dataTransfer?.files?.[0];
      if (file) setFile(file);
    });

    function handlePaste(event) {
      const items = event.clipboardData?.items;
      if (!items) return;

      for (const item of items) {
        if (item.type && item.type.startsWith("image/")) {
          event.preventDefault();
          const file = item.getAsFile();
          if (file) setFile(file);
          return;
        }
      }
    }

    dropzone.addEventListener("paste", handlePaste);

    document.addEventListener("paste", (event) => {
      if (!dropzone.isConnected) return;

      const active = document.activeElement;
      if (
        active &&
        (active.tagName === "INPUT" ||
          active.tagName === "TEXTAREA" ||
          active.isContentEditable) &&
        !dropzone.contains(active)
      ) {
        return;
      }

      handlePaste(event);
    });

    dropzone.querySelector('[data-action="browse"]')?.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      fileInput.click();
    });

    dropzone.querySelector('[data-action="camera"]')?.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      (cameraInput || fileInput).click();
    });

    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        const target = event.target.closest("[data-action]");
        if (!target) {
          event.preventDefault();
          fileInput.click();
        }
      }
    });

    return {
      getFile() {
        return selectedFile || fileInput.files[0] || null;
      },
      setFile,
    };
  };
})();
