/**
 * GlucoSense AI - Client-side Healthcare Application Script
 */

document.addEventListener('DOMContentLoaded', () => {
  // Preset Clinical Profiles based on verified dataset distributions
  const PRESETS = {
    low: {
      pregnancies: 1,
      glucose: 85,
      bloodpressure: 66,
      skinthickness: 29,
      insulin: 26,
      bmi: 22.5,
      dpf: 0.24,
      age: 25
    },
    high: {
      pregnancies: 5,
      glucose: 168,
      bloodpressure: 84,
      skinthickness: 36,
      insulin: 180,
      bmi: 35.8,
      dpf: 0.65,
      age: 52
    },
    borderline: {
      pregnancies: 2,
      glucose: 122,
      bloodpressure: 74,
      skinthickness: 26,
      insulin: 95,
      bmi: 28.4,
      dpf: 0.38,
      age: 38
    }
  };

  // Wire up preset buttons
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const type = btn.getAttribute('data-preset');
      const data = PRESETS[type];
      if (!data) return;

      Object.keys(data).forEach(key => {
        const input = document.querySelector(`input[name="${key}"]`);
        if (input) {
          input.value = data[key];
          validateInput(input);
        }
      });

      // Smooth scroll to form
      const formEl = document.getElementById('prediction-form');
      if (formEl) {
        formEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }

      // Visual feedback on preset button
      const originalText = btn.innerHTML;
      btn.style.borderColor = 'var(--accent-cyan)';
      setTimeout(() => {
        btn.style.borderColor = '';
      }, 1200);
    });
  });

  // Interactive BMI Calculator Drawer
  const bmiToggle = document.getElementById('toggle-bmi-calc');
  const bmiDrawer = document.getElementById('bmi-calc-drawer');
  const btnApplyBmi = document.getElementById('btn-apply-bmi');

  if (bmiToggle && bmiDrawer) {
    bmiToggle.addEventListener('click', (e) => {
      e.preventDefault();
      bmiDrawer.classList.toggle('active');
    });
  }

  if (btnApplyBmi) {
    btnApplyBmi.addEventListener('click', (e) => {
      e.preventDefault();
      const heightCm = parseFloat(document.getElementById('calc-height')?.value);
      const weightKg = parseFloat(document.getElementById('calc-weight')?.value);

      if (heightCm > 50 && heightCm < 260 && weightKg > 20 && weightKg < 350) {
        const heightM = heightCm / 100;
        const calculatedBmi = (weightKg / (heightM * heightM)).toFixed(1);
        const bmiInput = document.querySelector('input[name="bmi"]');
        if (bmiInput) {
          bmiInput.value = calculatedBmi;
          validateInput(bmiInput);
          bmiDrawer.classList.remove('active');
        }
      } else {
        alert('Please enter realistic height (50-260 cm) and weight (20-350 kg).');
      }
    });
  }

  // Real-time input validation & constraints
  // NOTE: only select inputs that have a `name` attribute (i.e. actual form
  // fields). The BMI-calculator helper inputs (#calc-height / #calc-weight)
  // live inside the form tag but must NOT be validated or block submission.
  const form = document.getElementById('prediction-form');
  const inputs = form
    ? form.querySelectorAll('input[type="number"][name], input[type="text"][name]')
    : [];

  function validateInput(input) {
    const val = parseFloat(input.value);
    const card = input.closest('.input-card');
    const errorMsg = card?.querySelector('.validation-feedback');
    const name = input.name;

    let isValid = true;
    let message = '';

    if (input.value.trim() === '') {
      isValid = false;
      message = 'This field is required.';
    } else if (isNaN(val)) {
      isValid = false;
      message = 'Please enter a valid number.';
    } else if (val < 0) {
      isValid = false;
      message = 'Cannot be negative.';
    } else {
      // Specific clinical bounds
      if (name === 'glucose' && (val < 40 || val > 400)) {
        isValid = false;
        message = 'Typical plasma glucose ranges between 40 and 400 mg/dL.';
      } else if (name === 'bloodpressure' && (val < 30 || val > 200)) {
        isValid = false;
        message = 'Diastolic BP is typically between 30 and 200 mmHg.';
      } else if (name === 'bmi' && (val < 10 || val > 75)) {
        isValid = false;
        message = 'Typical BMI ranges from 10 to 75 kg/m².';
      } else if (name === 'age' && (val < 1 || val > 120)) {
        isValid = false;
        message = 'Please enter a valid age between 1 and 120.';
      } else if (name === 'pregnancies' && val > 25) {
        isValid = false;
        message = 'Number of pregnancies exceeds expected range.';
      } else if (name === 'dpf' && val > 3.0) {
        isValid = false;
        message = 'Pedigree function typically ranges from 0.05 to 2.5.';
      }
    }

    if (!isValid) {
      // Show error state for ALL invalid fields (including empty ones)
      card?.classList.add('has-error');
      if (errorMsg) errorMsg.textContent = message;
    } else {
      card?.classList.remove('has-error');
      if (errorMsg) errorMsg.textContent = '';
    }

    return isValid;
  }

  inputs.forEach(input => {
    input.addEventListener('input', () => validateInput(input));
    input.addEventListener('blur', () => validateInput(input));
  });

  // Handle form submission with loading state
  if (form) {
    form.addEventListener('submit', (e) => {
      let formHasErrors = false;
      inputs.forEach(input => {
        if (!validateInput(input)) {
          formHasErrors = true;
        }
      });

      if (formHasErrors) {
        e.preventDefault();
        const firstError = form.querySelector('.has-error input');
        if (firstError) {
          firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
          firstError.focus();
        }
        return false;
      }

      // Show loading spinner on button — allow normal form POST to continue
      const submitBtn = form.querySelector('.btn-submit');
      if (submitBtn) {
        submitBtn.classList.add('is-loading');
        const textSpan = submitBtn.querySelector('.btn-text-content');
        if (textSpan) {
          textSpan.textContent = 'Running Model Inference...';
        }
      }
      // do NOT call e.preventDefault() — let the browser POST normally
    });

    // Reset button handler
    const resetBtn = document.getElementById('btn-reset-form');
    if (resetBtn) {
      resetBtn.addEventListener('click', (e) => {
        e.preventDefault();
        form.reset();
        document.querySelectorAll('.input-card').forEach(card => {
          card.classList.remove('has-error');
          const fb = card.querySelector('.validation-feedback');
          if (fb) fb.textContent = '';
        });
      });
    }
  }

  // Print Report Handler on Result Page
  const printBtn = document.getElementById('btn-print-report');
  if (printBtn) {
    printBtn.addEventListener('click', (e) => {
      e.preventDefault();
      window.print();
    });
  }

  // Gauge fill animation on Result Page
  const gaugeFill = document.querySelector('.gauge-fill[data-probability]');
  if (gaugeFill) {
    const prob = gaugeFill.getAttribute('data-probability');
    if (prob !== null && prob !== '') {
      gaugeFill.style.width = '0%';
      requestAnimationFrame(() => {
        setTimeout(() => {
          gaugeFill.style.width = prob + '%';
        }, 80);
      });
    }
  }
});
