// ================= GLOBAL =================

let file = null;

// ================= FILE UPLOAD =================

document.getElementById("fileInput").onchange = e => {
  file = e.target.files[0];
};

// ================= PAGE NAVIGATION =================

function goTo(n){
  document.querySelectorAll(".page")
    .forEach(p => p.classList.remove("active"));

  document.getElementById("page" + n)
    .classList.add("active");
}

// ================= CLINICAL INTERPRETATION =================

function interpretation(type, severity){

  if(type === "notumor"){
    return "No abnormal intracranial lesion suggestive of tumor is detected in the analyzed MRI scan. Brain structures appear within normal limits.";
  }

  if(type === "glioma"){
    if(severity === "Mild")
      return "A small localized glioma lesion is detected with minimal surrounding involvement, suggesting an early-stage condition.";
    if(severity === "Moderate")
      return "A moderately sized glioma is identified with noticeable spread into adjacent brain tissue. Clinical evaluation is recommended.";
    return "A large and potentially aggressive glioma with extensive involvement is detected. Immediate specialist consultation is advised.";
  }

  if(type === "meningioma"){
    if(severity === "Mild")
      return "A small extra-axial mass consistent with meningioma is observed. Features suggest a benign and slow-growing lesion.";
    if(severity === "Moderate")
      return "A moderately sized meningioma is present with mass effect on adjacent brain structures.";
    return "A large meningioma with significant compression of surrounding tissue is detected. Surgical assessment may be required.";
  }

  if(type === "pituitary"){
    if(severity === "Mild")
      return "A small pituitary lesion is identified, likely representing a microadenoma.";
    if(severity === "Moderate")
      return "A pituitary mass of moderate size is observed with potential impact on surrounding structures.";
    return "A large pituitary tumor is detected with possible compression of optic pathways. Urgent evaluation is recommended.";
  }

  return "";
}

// ================= RECOMMENDED ACTION =================

function recommendation(type){

  if(type === "notumor"){
    return "Routine clinical monitoring is suggested. No immediate intervention required unless symptoms develop.";
  }

  return "Further evaluation by a neurologist or neurosurgeon is advised. MRI with contrast and clinical correlation recommended.";
}

// ================= MAIN ANALYSIS =================

async function analyze(){

  if(!file){
    alert("Upload MRI first");
    return;
  }

  // Disable button during processing
  const btn = document.querySelector(".primary");
  btn.disabled = true;
  btn.textContent = "Analyzing...";

  try {

    // Show input preview
    inputImg.src = URL.createObjectURL(file);
    segInputImg.src = inputImg.src;

    // Send image to backend
    const fd = new FormData();
    fd.append("file", file);

    const res = await fetch("/predict", {
      method: "POST",
      body: fd
    });

    if(!res.ok){
      throw new Error("Server error");
    }

    const result = await res.json();

    // ================= DIAGNOSIS PAGE =================

    type.textContent = result.tumor_type;
    conf.textContent = result.confidence;
    area.textContent = result.tumor_percentage;
    severity.textContent = result.tumor_level;

    interp.textContent =
      interpretation(result.tumor_type, result.tumor_level);

    // ================= SEGMENTATION IMAGES =================

    maskImg.src =
      result.mask_image + "?t=" + Date.now();

    overlayImg.src =
      result.overlay_image + "?t=" + Date.now();

    // ================= CLINICAL REPORT PAGE =================

    r_name.textContent = name.value;
    r_age.textContent = age.value;
    r_pid.textContent = pid.value;

    r_type.textContent = result.tumor_type;
    r_conf.textContent = result.confidence;
    r_area.textContent = result.tumor_percentage;
    r_severity.textContent = result.tumor_level;

    r_desc.textContent =
      interpretation(result.tumor_type, result.tumor_level);

    r_action.textContent =
      recommendation(result.tumor_type);

    finalOverlay.src =
      result.overlay_image + "?t=" + Date.now();

    // Go to diagnosis page
    goTo(2);

  } catch(err){

    alert("Analysis failed. Please try another image.");
    console.error(err);

  } finally {

    btn.disabled = false;
    btn.textContent = "Begin AI Analysis →";

  }
}