// IBM Analytics — loaded via scripts: in docusaurus.config.ts

(function() {
  // Only run on the production site
  if (window.location.hostname !== 'docs.mellea.ai') {
    console.log('IBM Analytics disabled outside production');
    return;
  }

  // Set up IBM Analytics configuration
  window.idaPageIsSPA = true;

  // Configure digital data
  window.digitalData = {
    page: {
      category: {
        primaryCategory: 'PC340'
      },
      pageInfo: {
        ibm: {
          siteId: 'granite-developer-enablement'
        }
      }
    }
  };

  // Configure IBM Analytics settings
  window._ibmAnalytics = {
    settings: {
      name: 'granite-developer-enablement',
      isSpa: true
    }
  };

  // Load IBM Analytics library
  var script = document.createElement('script');
  script.src = 'https://1.www.s81c.com/common/stats/ibm-common.js';
  script.type = 'text/javascript';
  document.head.appendChild(script);

  // Cancellation token: a newer navigation supersedes any in-flight poll so
  // rapid navigations before ibmStats loads don't each fire a pageview.
  var currentPollId = 0;
  function trackPageview() {
    var myId = ++currentPollId;
    // Wait for IBM Analytics to load, then track pageview (poll up to ~2s)
    var attempts = 0;
    (function attempt() {
      if (myId !== currentPollId) return; // superseded by a newer navigation
      if (window.ibmStats && typeof window.ibmStats.pageview === 'function') {
        window.ibmStats.pageview();
      } else if (attempts++ < 20) {
        setTimeout(attempt, 100);
      }
    })();
  }

  // Track initial page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', trackPageview);
  } else {
    trackPageview();
  }

  function emitOnHistoryChange(type) {
    if (history[type].__analyticsPatched) return; // don't double-wrap if included twice
    var orig = history[type];
    history[type] = function() {
      var ret = orig.apply(this, arguments);
      trackPageview();
      return ret;
    };
    history[type].__analyticsPatched = true;
  }
  if (window.navigation) {
    // Skip in-page #anchor jumps; only track real path/query changes.
    window.navigation.addEventListener("navigate", function(event) {
      var to = new URL(event.destination.url);
      if (to.pathname === location.pathname && to.search === location.search) return;
      trackPageview();
    });
  } else {
    emitOnHistoryChange('pushState');
    emitOnHistoryChange('replaceState');
    window.addEventListener('popstate', trackPageview);
  }
})();
