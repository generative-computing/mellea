// ==========================================================================
// LANDING PAGE INTERACTIVE BEHAVIOR
// Uses event delegation on document so it works with React-rendered DOM.
// ==========================================================================

// ==========================================================================
// CODE SNIPPETS DATA
// ==========================================================================
const CODE_SNIPPETS = {
  'instruct-validate-repair': `import mellea
from mellea.stdlib.sampling import RejectionSamplingStrategy


def write_email_with_strategy(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email_candidate = m.instruct(
        f"Write an email to {name} using the notes following: {notes}.",
        requirements=[
            "The email should have a salutation.",
            "Use a formal tone.",
        ],
        strategy=RejectionSamplingStrategy(loop_budget=3),
        return_sampling_results=True,
    )

    if email_candidate.success:
        return str(email_candidate.result)

    # If sampling fails, use the first generation
    print("Expect sub-par result.")
    return email_candidate.sample_generations[0].value`,

  'generative-slots': `@mellea.generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
  """Classify the sentiment of the input text as 'positive' or 'negative'."""

sentiment = classify_sentiment(m, text=customer_review)

if sentiment == "positive":
    msg = m.instruct("Thank the customer for their post")
else:
    msg = m.instruct(
       description="Apologize for the customer's negative experience and offer a 5% discount for their next visit",
       grounding_context={"review": customer_review}
    )

post_response(msg)`,

  'mobjects': `import mellea
from mellea.stdlib.mify import mify, MifiedProtocol
import pandas
from io import StringIO


@mify(fields_include={"table"}, template="{{ table }}")
class MyCompanyDatabase:
  table: str = """| Store      | Sales   |
                    | ---------- | ------- |
                    | Northeast  | $250    |
                    | Southeast  | $80     |
                    | Midwest    | $420    |"""

  def transpose(self):
    pandas.read_csv(
      StringIO(self.table),
      sep='|',
      skipinitialspace=True,
      header=0,
      index_col=False
    )


m = mellea.start_session()
db = MyCompanyDatabase()
assert isinstance(db, MifiedProtocol)
answer = m.query(db, "What were sales for the Northeast branch this month?")
print(str(answer))`
};

// ==========================================================================
// EVENT DELEGATION — single click handler on document
// ==========================================================================
document.addEventListener('click', function(e) {
  // --- Copy text button (e.g. "uv pip install mellea") ---
  var copyBtn = e.target.closest('[data-copy-text]');
  if (copyBtn) {
    var text = copyBtn.getAttribute('data-copy-text');
    navigator.clipboard.writeText(text).then(function() {
      copyBtn.classList.add('copied');
      var span = copyBtn.querySelector('span');
      var originalText = span.textContent;
      span.textContent = 'Copied!';
      setTimeout(function() {
        copyBtn.classList.remove('copied');
        span.textContent = originalText;
      }, 2000);
    });
    return;
  }

  // --- Copy code block button ---
  var codeCopyBtn = e.target.closest('[data-copy-code]');
  if (codeCopyBtn) {
    var codeBlock = codeCopyBtn.closest('.landing-code-block');
    var code = codeBlock.querySelector('code');
    navigator.clipboard.writeText(code.textContent).then(function() {
      codeCopyBtn.classList.add('copied');
      setTimeout(function() {
        codeCopyBtn.classList.remove('copied');
      }, 2000);
    });
    return;
  }

  // --- Comparison slider: click to jump ---
  var slider = e.target.closest('.comparison-slider');
  if (slider && !e.target.closest('.comparison-handle')) {
    updateSliderPosition(slider, e.clientX);
    return;
  }

  // --- Accordion headers ---
  var accHeader = e.target.closest('.landing-accordion-header');
  if (accHeader) {
    var item = accHeader.closest('.landing-accordion-item');
    if (!item || item.classList.contains('active')) return;

    var accordion = item.closest('.landing-accordion');
    var desktopSnippet = document.getElementById('snippet-desktop');
    var desktopCode = document.getElementById('desktop-code');

    // Fade out desktop snippet
    if (desktopSnippet) {
      desktopSnippet.classList.add('transitioning');
    }

    setTimeout(function() {
      // Switch active item
      accordion.querySelectorAll('.landing-accordion-item').forEach(function(i) {
        i.classList.remove('active');
      });
      item.classList.add('active');

      // Update desktop code panel
      if (desktopCode && typeof hljs !== 'undefined') {
        var snippetKey = item.getAttribute('data-snippet');
        var highlighted = hljs.highlight(CODE_SNIPPETS[snippetKey], { language: 'python' });
        desktopCode.innerHTML = highlighted.value;
        desktopCode.className = 'language-python hljs';
      }

      // Fade in
      if (desktopSnippet) {
        setTimeout(function() {
          desktopSnippet.classList.remove('transitioning');
        }, 50);
      }
    }, 150);
    return;
  }
});

// ==========================================================================
// COMPARISON SLIDER — drag handling
// ==========================================================================
var sliderDragging = false;

function updateSliderPosition(slider, clientX) {
  var afterImage = slider.querySelector('.after-image');
  var handle = slider.querySelector('.comparison-handle');
  if (!afterImage || !handle) return;

  var rect = slider.getBoundingClientRect();
  var pos = ((clientX - rect.left) / rect.width) * 100;
  pos = Math.max(0, Math.min(100, pos));

  afterImage.style.clipPath = 'inset(0 0 0 ' + pos + '%)';
  handle.style.left = pos + '%';
}

document.addEventListener('mousedown', function(e) {
  if (e.target.closest('.comparison-handle')) {
    sliderDragging = true;
    e.preventDefault();
  }
});

document.addEventListener('touchstart', function(e) {
  if (e.target.closest('.comparison-handle')) {
    sliderDragging = true;
  }
}, { passive: true });

document.addEventListener('mousemove', function(e) {
  if (!sliderDragging) return;
  var slider = document.getElementById('comparison-slider');
  if (slider) updateSliderPosition(slider, e.clientX);
});

document.addEventListener('touchmove', function(e) {
  if (!sliderDragging) return;
  var slider = document.getElementById('comparison-slider');
  if (slider && e.touches[0]) updateSliderPosition(slider, e.touches[0].clientX);
}, { passive: true });

document.addEventListener('mouseup', function() {
  sliderDragging = false;
});

document.addEventListener('touchend', function() {
  sliderDragging = false;
});

// ==========================================================================
// LOAD HIGHLIGHT.JS & INITIALIZE
// ==========================================================================
(function loadHighlightJS() {
  var hljsLink = document.createElement('link');
  hljsLink.rel = 'stylesheet';
  hljsLink.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
  document.head.appendChild(hljsLink);

  var hljsScript = document.createElement('script');
  hljsScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
  hljsScript.onload = function() {
    var pythonScript = document.createElement('script');
    pythonScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js';
    pythonScript.onload = function() {
      hljs.highlightAll();
    };
    document.head.appendChild(pythonScript);
  };
  document.head.appendChild(hljsScript);
})();
