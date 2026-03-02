(function () {
  const API = "/api";

  function getMain() {
    return document.getElementById("main");
  }

  function parseHash() {
    const hash = window.location.hash.slice(1) || "/";
    const parts = hash.split("/").filter(Boolean);
    if (parts[0] === "experiment" && parts[1]) {
      return { view: "detail", name: parts[1] };
    }
    if (parts[0] === "new") return { view: "new" };
    return { view: "list" };
  }

  function renderList(sortBy) {
    sortBy = sortBy || "name";
    getMain().innerHTML = '<p class="loading">Loading experiments…</p>';
    fetch(API + "/experiments?sort=" + encodeURIComponent(sortBy), { cache: "no-store" })
      .then((r) => r.json())
      .then((items) => {
        if (items.length === 0) {
          getMain().innerHTML =
            '<p class="page-title">Experiments</p><p class="loading">No experiments yet. Create one from "New experiment".</p>';
          return;
        }
        const toolbar = document.createElement("div");
        toolbar.className = "list-toolbar";
        toolbar.innerHTML =
          'Sort: <select id="sortSelect">' +
          '<option value="time"' + (sortBy === "time" ? " selected" : "") + '>Newest first</option>' +
          '<option value="time_asc"' + (sortBy === "time_asc" ? " selected" : "") + '>Oldest first</option>' +
          '<option value="name"' + (sortBy === "name" ? " selected" : "") + '>Name A–Z</option>' +
          '<option value="name_desc"' + (sortBy === "name_desc" ? " selected" : "") + '>Name Z–A</option>' +
          "</select>";
        const table = document.createElement("table");
        table.className = "exp-table";
        table.innerHTML =
          "<thead><tr>" +
          "<th>Name</th>" +
          "<th>Checkpoints</th>" +
          "<th>Dataset</th>" +
          "<th>Dataset count</th>" +
          "<th>Modified</th>" +
          "</tr></thead><tbody></tbody>";
        const tbody = table.querySelector("tbody");
        items.forEach((exp) => {
          const mtime = new Date(exp.mtime * 1000).toLocaleString();
          const preview = (exp.dataset_preview || []).slice(0, 3);
          let datasetCell = "";
          if (preview.length > 0) {
            datasetCell = '<div class="exp-table-dataset-preview">' +
              preview.map(function (fname) {
                return '<img src="' + API + "/experiments/" + encodeURIComponent(exp.name) + "/dataset/" + encodeURIComponent(fname) + '" alt="" class="exp-table-dataset-img">';
              }).join("") +
              "</div>";
          } else {
            datasetCell = "—";
          }
          const datasetCount = (exp.dataset_count != null && exp.dataset_count > 0) ? String(exp.dataset_count) : "—";
          const row = document.createElement("tr");
          row.innerHTML =
            "<td>" +
            '<a href="#/experiment/' + encodeURIComponent(exp.name) + '" class="exp-table-link">' + escapeHtml(exp.name) + "</a>" +
            "</td>" +
            "<td>" + escapeHtml(String(exp.n_checkpoints)) + "</td>" +
            "<td>" + datasetCell + "</td>" +
            "<td>" + escapeHtml(datasetCount) + "</td>" +
            "<td>" + escapeHtml(mtime) + "</td>";
          tbody.appendChild(row);
        });
        getMain().innerHTML = "<p class=\"page-title\">Experiments</p>";
        getMain().appendChild(toolbar);
        getMain().appendChild(table);
        var sortSelect = getMain().querySelector("#sortSelect");
        if (sortSelect) {
          sortSelect.addEventListener("change", function () {
            renderList(this.value);
          });
        }
      })
      .catch((err) => {
        getMain().innerHTML =
          '<p class="message error">Failed to load experiments: ' +
          escapeHtml(String(err)) +
          "</p>";
      });
  }

  function renderDetail(name) {
    getMain().innerHTML = '<p class="loading">Loading…</p>';
    const back =
      '<a href="#/" class="back-link">← Back to list</a>';
    const title =
      '<p class="page-title">Experiment: ' + escapeHtml(name) + '</p>';

    const statusEl = document.createElement("p");
    statusEl.className = "loading";
    statusEl.innerHTML = "Checking status…";

    const tabBar = document.createElement("div");
    tabBar.className = "detail-tabs";
    tabBar.innerHTML =
      '<button type="button" class="tab-btn active" data-tab="log">Log</button>' +
      '<button type="button" class="tab-btn" data-tab="dataset">Dataset</button>' +
      '<button type="button" class="tab-btn" data-tab="images">Images</button>' +
      '<button type="button" class="tab-btn" data-tab="generated_img">Generated</button>' +
      '<button type="button" class="tab-btn" data-tab="checkpoints">Checkpoints</button>';

    const logPanel = document.createElement("div");
    logPanel.className = "tab-panel active";
    logPanel.setAttribute("data-tab", "log");
    logPanel.innerHTML =
      '<div class="metrics-section">' +
      '<h3 class="metrics-title">Metrics</h3>' +
      '<div class="chart-section">' +
      '<div class="chart-controls"><span class="chart-label">Chart columns:</span> <div id="chartColumnSelect" class="chart-column-select"></div> <button type="button" class="btn btn-small" id="chartClearAll">Clear all</button></div>' +
      '<div class="chart-wrap"><canvas id="metricsChart"></canvas></div>' +
      '</div>' +
      '<div id="metricsTableWrap" class="metrics-table-wrap">(loading…)</div>' +
      '</div>' +
      '<div class="log-raw-section">' +
      '<button type="button" class="log-raw-toggle collapsed" id="logRawToggle" aria-expanded="false">Raw log ▼</button>' +
      '<div class="log-raw-content" id="logRawContent">' +
      '<div class="log-toolbar">' +
      '<label>Tail <input type="number" id="tailLines" value="500" min="1" max="10000"></label> ' +
      '<button class="btn" id="refreshLog">Refresh</button>' +
      "</div>" +
      '<div class="log-container"><pre id="logContent"></pre></div>' +
      '</div></div>';

    const datasetPanel = document.createElement("div");
    datasetPanel.className = "tab-panel";
    datasetPanel.setAttribute("data-tab", "dataset");
    datasetPanel.innerHTML = '<div class="img-grid dataset-grid" id="datasetGrid"></div>';

    const imgPanel = document.createElement("div");
    imgPanel.className = "tab-panel";
    imgPanel.setAttribute("data-tab", "images");
    imgPanel.innerHTML = '<div class="img-grid" id="imgGrid"></div>';

    const generatedImgPanel = document.createElement("div");
    generatedImgPanel.className = "tab-panel";
    generatedImgPanel.setAttribute("data-tab", "generated_img");
    generatedImgPanel.innerHTML =
      '<div class="generate-form-section">' +
      '<button type="button" class="generate-toggle collapsed" id="generateToggle" aria-expanded="false">Command ▼</button>' +
      '<div class="generate-form-wrap" id="generateFormWrap">' +
      '<div class="generate-form">' +
      '<div class="form-row"><label for="genSourceFont">Source font</label><select id="genSourceFont"><option value="">-- select --</option></select></div>' +
      '<div class="form-row"><label for="genCharFile">Char file</label><select id="genCharFile"><option value="">-- select --</option></select></div>' +
      '<div class="form-row"><label for="genCheckpoint">Checkpoint</label><select id="genCheckpoint"></select></div>' +
      '<div class="form-row"><label class="gen-command-label">Command <button type="button" class="btn btn-small gen-copy-btn" id="genCopyBtn">Copy</button></label><pre id="genCommand" class="gen-command"></pre></div>' +
      '</div></div></div>' +
      '<div class="generated-img-wrapper"><div class="img-grid dataset-grid" id="generatedImgGrid"></div><div id="generatedImgLoadMoreWrap" class="generated-load-more-wrap"></div></div>';

    const ckptPanel = document.createElement("div");
    ckptPanel.className = "tab-panel";
    ckptPanel.setAttribute("data-tab", "checkpoints");
    ckptPanel.innerHTML = '<div class="ckpt-list" id="ckptList"></div>';

    getMain().innerHTML = "";
    getMain().appendChild(document.createRange().createContextualFragment(back));
    getMain().appendChild(document.createRange().createContextualFragment(title));
    getMain().appendChild(statusEl);
    getMain().appendChild(tabBar);
    getMain().appendChild(logPanel);
    getMain().appendChild(datasetPanel);
    getMain().appendChild(imgPanel);
    getMain().appendChild(generatedImgPanel);
    getMain().appendChild(ckptPanel);

    tabBar.querySelectorAll(".tab-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const tab = this.getAttribute("data-tab");
        tabBar.querySelectorAll(".tab-btn").forEach(function (b) { b.classList.remove("active"); });
        getMain().querySelectorAll(".tab-panel").forEach(function (p) {
          p.classList.toggle("active", p.getAttribute("data-tab") === tab);
        });
        this.classList.add("active");
        if (tab === "log" && metricsChart) {
          setTimeout(function () { metricsChart.resize(); }, 50);
        }
        if (tab === "generated_img" && typeof populateGenForm === "function") {
          populateGenForm();
        }
      });
    });

    function loadStatus() {
      fetch(API + "/experiments/" + encodeURIComponent(name) + "/status")
        .then((r) => r.json())
        .then((d) => {
          if (d.status === "running") {
            statusEl.innerHTML =
              'Status: <span class="status-badge running">Running</span>';
          } else {
            statusEl.innerHTML =
              'Status: <span class="status-badge finished">Finished</span>';
          }
        })
        .catch(() => {
          statusEl.innerHTML = "Status: unknown";
        });
    }

    function loadLog() {
      const tail = document.getElementById("tailLines");
      const n = tail ? parseInt(tail.value, 10) : 500;
      const url =
        API +
        "/experiments/" +
        encodeURIComponent(name) +
        "/log?tail=" +
        (n > 0 ? n : 500);
      fetch(url)
        .then((r) => r.json())
        .then((d) => {
          const pre = document.getElementById("logContent");
          if (pre) pre.textContent = d.content || "(empty)";
        })
        .catch(() => {
          const pre = document.getElementById("logContent");
          if (pre) pre.textContent = "(failed to load log)";
        });
    }

    var metricsChart = null;
    var metricsBlocks = [];

    function toChartValue(v) {
      if (v === undefined || v === null) return null;
      if (typeof v === "number" && !isNaN(v)) return v;
      if (typeof v === "string") {
        if (v.endsWith("%")) return parseFloat(v) || null;
        var n = parseFloat(v);
        return isNaN(n) ? null : n;
      }
      return null;
    }

    function isChartable(blocks, key) {
      if (key === "step" || key === "timestamp") return false;
      for (var i = 0; i < blocks.length; i++) {
        var v = toChartValue(blocks[i][key]);
        if (v !== null) return true;
      }
      return false;
    }

    function updateChart(blocks, selectedKeys) {
      if (typeof Chart === "undefined") return;
      var canvas = document.getElementById("metricsChart");
      if (!canvas) return;
      if (!selectedKeys || selectedKeys.length === 0) {
        if (metricsChart) { metricsChart.destroy(); metricsChart = null; }
        return;
      }
      var steps = blocks.map(function (b) { return b.step; });
      var datasets = selectedKeys.map(function (key, i) {
        var colors = ["#67c6c0", "#f59e0b", "#8b5cf6", "#ef4444", "#22c55e", "#06b6d4", "#ec4899", "#84cc16"];
        var color = colors[i % colors.length];
        return {
          label: key,
          data: blocks.map(function (b) { return toChartValue(b[key]); }),
          borderColor: color,
          backgroundColor: color + "20",
          fill: false,
          tension: 0.1,
        };
      });
      if (metricsChart) metricsChart.destroy();
      metricsChart = new Chart(canvas, {
        type: "line",
        data: { labels: steps, datasets: datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { intersect: false, mode: "index" },
          plugins: {
            legend: { position: "top" },
            tooltip: {
              enabled: true,
              mode: "index",
              intersect: false,
              backgroundColor: "rgba(37, 37, 41, 0.95)",
              titleColor: "#e4e4e7",
              bodyColor: "#d4d4d8",
              borderColor: "#3f3f46",
              borderWidth: 1,
              padding: 10,
              callbacks: {
                title: function (items) {
                  return "Step " + (items[0] && items[0].label != null ? items[0].label : "");
                },
                label: function (ctx) {
                  var v = ctx.parsed.y;
                  var s = v != null && !isNaN(v) ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : "-";
                  return ctx.dataset.label + ": " + s;
                },
              },
            },
          },
          scales: {
            x: { title: { display: true, text: "Step" } },
            y: { beginAtZero: false },
          },
        },
      });
    }

    function loadMetrics() {
      const wrap = document.getElementById("metricsTableWrap");
      const selectWrap = document.getElementById("chartColumnSelect");
      if (!wrap) return;
      const base = API + "/experiments/" + encodeURIComponent(name);
      fetch(base + "/metrics")
        .then(function (r) {
          if (r.status === 404) {
            return fetch(base + "/parse-log", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
              .then(function (pr) {
                if (!pr.ok) return pr.json().then(function (d) { throw new Error(d.error || pr.status); });
                return fetch(base + "/metrics").then(function (rr) {
                  if (!rr.ok) throw new Error("Failed to load metrics after parse");
                  return rr.json();
                });
              });
          }
          if (!r.ok) throw new Error(r.statusText);
          return r.json();
        })
        .then(function (blocks) {
          metricsBlocks = blocks;
          if (!Array.isArray(blocks) || blocks.length === 0) {
            wrap.innerHTML = "<p class=\"metrics-empty\">No metrics yet.</p>";
            if (selectWrap) selectWrap.innerHTML = "";
            return;
          }
          var keys = [];
          var seen = {};
          blocks.forEach(function (b) {
            Object.keys(b).forEach(function (k) {
              if (!seen[k]) { seen[k] = true; keys.push(k); }
            });
          });
          keys.sort(function (a, b) {
            if (a === "step") return -1;
            if (b === "step") return 1;
            if (a === "timestamp") return -1;
            if (b === "timestamp") return 1;
            return a.localeCompare(b);
          });
          var chartableKeys = keys.filter(function (k) { return isChartable(blocks, k); });
          if (selectWrap) {
            selectWrap.innerHTML = "";
            var defaults = ["AC_acc_c", "AC_acc_s", "AC_g_acc_c", "AC_g_acc_s"];
            chartableKeys.forEach(function (k) {
              var label = document.createElement("label");
              label.className = "chart-check";
              var cb = document.createElement("input");
              cb.type = "checkbox";
              cb.dataset.key = k;
              if (defaults.indexOf(k) >= 0) cb.checked = true;
              cb.addEventListener("change", function () {
                var sel = Array.from(selectWrap.querySelectorAll("input:checked")).map(function (x) { return x.dataset.key; });
                updateChart(metricsBlocks, sel);
              });
              label.appendChild(cb);
              label.appendChild(document.createTextNode(" " + k));
              selectWrap.appendChild(label);
            });
            var sel = Array.from(selectWrap.querySelectorAll("input:checked")).map(function (x) { return x.dataset.key; });
            if (sel.length) updateChart(metricsBlocks, sel);
            var clearBtn = document.getElementById("chartClearAll");
            if (clearBtn) {
              clearBtn.onclick = function () {
                selectWrap.querySelectorAll("input[type=checkbox]").forEach(function (cb) { cb.checked = false; });
                updateChart(metricsBlocks, []);
              };
            }
          }
          var html = "<table class=\"metrics-table\"><thead><tr>";
          keys.forEach(function (k) {
            html += "<th>" + escapeHtml(k) + "</th>";
          });
          html += "</tr></thead><tbody>";
          blocks.forEach(function (row) {
            html += "<tr>";
            keys.forEach(function (k) {
              var v = row[k];
              html += "<td>" + escapeHtml(v === undefined || v === null ? "" : String(v)) + "</td>";
            });
            html += "</tr>";
          });
          html += "</tbody></table>";
          wrap.innerHTML = html;
        })
        .catch(function (err) {
          wrap.innerHTML = "<p class=\"metrics-error\">" + escapeHtml(err.message || "Failed to load metrics") + "</p>";
          if (selectWrap) selectWrap.innerHTML = "";
        });
    }

    function loadCheckpoints() {
      const el = document.getElementById("ckptList");
      if (!el) return;
      fetch(API + "/experiments/" + encodeURIComponent(name) + "/checkpoints")
        .then(function (r) {
          if (!r.ok) throw new Error(r.status === 404 ? "Experiment not found" : "Failed to load");
          return r.json();
        })
        .then(function (list) {
          if (!Array.isArray(list) || list.length === 0) {
            el.textContent = "No checkpoints.";
            return;
          }
          el.textContent = "";
          list.forEach(function (f) {
            const a = document.createElement("a");
            a.href =
              API +
              "/experiments/" +
              encodeURIComponent(name) +
              "/checkpoints/" +
              encodeURIComponent(f.name);
            a.textContent = f.name;
            a.setAttribute("download", f.name);
            const wrap = document.createElement("div");
            wrap.className = "ckpt-item";
            wrap.appendChild(a);
            if (f.size != null && !isNaN(f.size)) {
              const sizeSpan = document.createElement("span");
              sizeSpan.className = "ckpt-size";
              sizeSpan.textContent = " (" + (f.size < 1024 ? f.size + " B" : (f.size / 1024 / 1024).toFixed(2) + " MB") + ")";
              wrap.appendChild(sizeSpan);
            }
            el.appendChild(wrap);
          });
        })
        .catch(function () {
          el.textContent = "Failed to load checkpoints.";
        });
    }

    function loadImages() {
      fetch(API + "/experiments/" + encodeURIComponent(name) + "/images")
        .then((r) => r.json())
        .then((list) => {
          const el = document.getElementById("imgGrid");
          if (!list.length) {
            el.textContent = "No images.";
            return;
          }
          list.slice().reverse().forEach((f) => {
            const wrap = document.createElement("div");
            wrap.className = "img-item";
            const a = document.createElement("a");
            a.href =
              API +
              "/experiments/" +
              encodeURIComponent(name) +
              "/images/" +
              encodeURIComponent(f.name);
            a.target = "_blank";
            const img = document.createElement("img");
            img.src = a.href;
            img.alt = f.name;
            a.appendChild(img);
            const caption = document.createElement("span");
            caption.className = "img-caption";
            caption.textContent = f.name;
            wrap.appendChild(a);
            wrap.appendChild(caption);
            el.appendChild(wrap);
          });
        })
        .catch(() => {
          const el = document.getElementById("imgGrid");
          if (el) el.textContent = "Failed to load images.";
        });
    }

    function loadDataset() {
      fetch(API + "/experiments/" + encodeURIComponent(name) + "/dataset")
        .then((r) => r.json())
        .then((list) => {
          const el = document.getElementById("datasetGrid");
          if (!list.length) {
            el.textContent = "No dataset images.";
            return;
          }
          list.forEach((f) => {
            const wrap = document.createElement("div");
            wrap.className = "img-item";
            const a = document.createElement("a");
            a.href =
              API +
              "/experiments/" +
              encodeURIComponent(name) +
              "/dataset/" +
              encodeURIComponent(f.name);
            a.target = "_blank";
            const img = document.createElement("img");
            img.src = a.href;
            img.alt = f.name;
            a.appendChild(img);
            const caption = document.createElement("span");
            caption.className = "img-caption";
            caption.textContent = f.name.replace(/\.png$/i, "");
            wrap.appendChild(a);
            wrap.appendChild(caption);
            el.appendChild(wrap);
          });
        })
        .catch(() => {
          const el = document.getElementById("datasetGrid");
          if (el) el.textContent = "Failed to load dataset images.";
        });
    }

    const INITIAL_GENERATED_COUNT = 200;
    const LOAD_MORE_GENERATED_COUNT = 200;

    function populateGenForm() {
      var ckptSel = document.getElementById("genCheckpoint");
      var fontSel = document.getElementById("genSourceFont");
      var charSel = document.getElementById("genCharFile");
      if (ckptSel) {
        fetch(API + "/experiments/" + encodeURIComponent(name) + "/checkpoints")
          .then(function (r) { return r.ok ? r.json() : []; })
          .then(function (list) {
            ckptSel.innerHTML = '<option value="">-- select --</option>';
            (list || []).forEach(function (f) {
              var opt = document.createElement("option");
              opt.value = f.name;
              opt.textContent = f.name;
              ckptSel.appendChild(opt);
            });
            updateGenCommand();
          })
          .catch(function () { ckptSel.innerHTML = '<option value="">-- select --</option>'; updateGenCommand(); });
      }
      if (fontSel) {
        fetch(API + "/source-fonts")
          .then(function (r) { return r.ok ? r.json() : []; })
          .then(function (list) {
            fontSel.innerHTML = '<option value="">-- select --</option>';
            (list || []).forEach(function (f) {
              var opt = document.createElement("option");
              opt.value = f.path;
              opt.textContent = f.name;
              fontSel.appendChild(opt);
            });
            updateGenCommand();
          })
          .catch(function () { updateGenCommand(); });
      }
      if (charSel) {
        fetch(API + "/cjk-ranges")
          .then(function (r) { return r.ok ? r.json() : []; })
          .then(function (list) {
            charSel.innerHTML = '<option value="">-- select --</option>';
            (list || []).forEach(function (f) {
              var opt = document.createElement("option");
              opt.value = f.path;
              opt.textContent = f.name;
              charSel.appendChild(opt);
            });
            updateGenCommand();
          })
          .catch(function () { updateGenCommand(); });
      }
    }

    function updateGenCommand() {
      var pre = document.getElementById("genCommand");
      var fontSel = document.getElementById("genSourceFont");
      var charSel = document.getElementById("genCharFile");
      var ckptSel = document.getElementById("genCheckpoint");
      if (!pre) return;
      var sourceFont = fontSel ? (fontSel.value || "").trim() : "";
      var charFile = charSel ? (charSel.value || "").trim() : "";
      var checkpoint = ckptSel ? (ckptSel.value || "").trim() : "";
      if (!sourceFont || !charFile || !checkpoint) {
        pre.textContent = "(select source font, char file, and checkpoint)";
        return;
      }
      var q = "source_font=" + encodeURIComponent(sourceFont) + "&char_file=" + encodeURIComponent(charFile) + "&checkpoint=" + encodeURIComponent(checkpoint);
      fetch(API + "/experiments/" + encodeURIComponent(name) + "/generate-command?" + q)
        .then(function (r) { return r.json(); })
        .then(function (d) {
          pre.textContent = d.error || d.command || "(error)";
        })
        .catch(function () { pre.textContent = "(failed to load command)"; });
    }

    function loadGeneratedImg() {
      fetch(API + "/experiments/" + encodeURIComponent(name) + "/generated_img")
        .then((r) => r.json())
        .then((list) => {
          const el = document.getElementById("generatedImgGrid");
          const loadMoreWrap = document.getElementById("generatedImgLoadMoreWrap");
          if (!el) return;
          if (!list.length) {
            el.textContent = "No generated images. Run model_comparsion.py with --output-dir exp/" + escapeHtml(name) + "/generated_img to add images.";
            if (loadMoreWrap) loadMoreWrap.innerHTML = "";
            return;
          }
          var displayedCount = 0;

          function appendImages(startIndex, count) {
            var endIndex = Math.min(startIndex + count, list.length);
            for (var i = startIndex; i < endIndex; i++) {
              var f = list[i];
              var wrap = document.createElement("div");
              wrap.className = "img-item";
              var a = document.createElement("a");
              a.href =
                API +
                "/experiments/" +
                encodeURIComponent(name) +
                "/generated_img/" +
                encodeURIComponent(f.name);
              a.target = "_blank";
              var img = document.createElement("img");
              img.src = a.href;
              img.alt = f.name;
              a.appendChild(img);
              var caption = document.createElement("span");
              caption.className = "img-caption";
              caption.textContent = f.name;
              wrap.appendChild(a);
              wrap.appendChild(caption);
              el.appendChild(wrap);
            }
            displayedCount = endIndex;
          }

          function updateLoadMoreButton() {
            if (!loadMoreWrap) return;
            if (displayedCount >= list.length) {
              loadMoreWrap.innerHTML = "";
              return;
            }
            var remaining = list.length - displayedCount;
            var loadCount = Math.min(LOAD_MORE_GENERATED_COUNT, remaining);
            loadMoreWrap.innerHTML = '<button type="button" class="btn btn-load-more" id="generatedImgLoadMoreBtn">Load more (' + loadCount + ' of ' + remaining + ' remaining)</button>';
            loadMoreWrap.querySelector("#generatedImgLoadMoreBtn").addEventListener("click", function () {
              appendImages(displayedCount, LOAD_MORE_GENERATED_COUNT);
              updateLoadMoreButton();
            });
          }

          appendImages(0, INITIAL_GENERATED_COUNT);
          updateLoadMoreButton();
        })
        .catch(() => {
          const el = document.getElementById("generatedImgGrid");
          if (el) el.textContent = "Failed to load generated images.";
          var loadMoreWrap = document.getElementById("generatedImgLoadMoreWrap");
          if (loadMoreWrap) loadMoreWrap.innerHTML = "";
        });
    }

    loadStatus();
    loadLog();
    loadMetrics();
    loadCheckpoints();
    loadImages();
    loadDataset();
    loadGeneratedImg();

    const refreshBtn = document.getElementById("refreshLog");
    if (refreshBtn) refreshBtn.addEventListener("click", function () { loadLog(); loadMetrics(); });

    const logRawToggle = document.getElementById("logRawToggle");
    const logRawContent = document.getElementById("logRawContent");
    if (logRawToggle && logRawContent) {
      logRawToggle.addEventListener("click", function () {
        var open = logRawContent.classList.toggle("open");
        logRawToggle.classList.toggle("collapsed", !open);
        logRawToggle.setAttribute("aria-expanded", open);
        logRawToggle.textContent = open ? "Raw log ▲" : "Raw log ▼";
      });
    }

    var generateToggle = document.getElementById("generateToggle");
    var generateFormWrap = document.getElementById("generateFormWrap");
    if (generateToggle && generateFormWrap) {
      generateToggle.addEventListener("click", function () {
        var open = generateFormWrap.classList.toggle("open");
        generateToggle.classList.toggle("collapsed", !open);
        generateToggle.setAttribute("aria-expanded", open);
        generateToggle.textContent = open ? "Command ▲" : "Command ▼";
        if (open) populateGenForm();
      });
    }
    ["genSourceFont", "genCharFile", "genCheckpoint"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.addEventListener("change", updateGenCommand);
    });
    var genCopyBtn = document.getElementById("genCopyBtn");
    if (genCopyBtn) {
      genCopyBtn.addEventListener("click", function () {
        var pre = document.getElementById("genCommand");
        var text = pre ? pre.textContent : "";
        if (!text || text.startsWith("(")) return;
        navigator.clipboard.writeText(text).then(function () {
          var orig = genCopyBtn.textContent;
          genCopyBtn.textContent = "Copied!";
          setTimeout(function () { genCopyBtn.textContent = orig; }, 1500);
        }).catch(function () {});
      });
    }
  }

  function renderNew() {
    const formHtml =
      '<a href="#/" class="back-link">← Back to list</a>' +
      '<p class="page-title">New experiment</p>' +
      '<form id="newExpForm" class="form">' +
      '<div class="form-group">' +
      '<label for="exp_name">Experiment name *</label>' +
      '<input type="text" id="exp_name" name="exp_name" required placeholder="e.g. my_font_1">' +
      '<span class="hint">Letters, numbers, underscore, hyphen only.</span>' +
      "</div>" +
      '<div class="form-group">' +
      '<label for="dataset_files">Dataset images (optional)</label>' +
      '<input type="file" id="dataset_files" name="dataset_files" accept=".png,.jpg,.jpeg,image/png,image/jpeg" multiple>' +
      '<span class="hint">Upload images to exp/&lt;name&gt;/dataset/. PNG, JPG, JPEG.</span>' +
      "</div>" +
      '<div class="form-actions">' +
      '<button type="submit" class="btn btn-primary">Create</button>' +
      "</div>" +
      "</form>" +
      '<div id="formMessage"></div>';

    getMain().innerHTML = formHtml;

    document.getElementById("newExpForm").addEventListener("submit", function (e) {
      e.preventDefault();
      const msgEl = document.getElementById("formMessage");
      msgEl.innerHTML = "";
      msgEl.className = "";

      const exp_name = document.getElementById("exp_name").value.trim();
      const fileInput = document.getElementById("dataset_files");
      const files = fileInput && fileInput.files ? Array.from(fileInput.files) : [];

      fetch(API + "/experiments/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exp_name: exp_name }),
      })
        .then((r) => {
          if (!r.ok) return r.json().then((d) => Promise.reject(d.error || r.statusText));
          return r.json();
        })
        .then(function (d) {
          const jobId = d.job_id;
          if (files.length === 0) {
            showCreateSuccess(msgEl, d, null);
            return;
          }
          const formData = new FormData();
          files.forEach(function (file) {
            formData.append("images", file);
          });
          return fetch(API + "/experiments/" + encodeURIComponent(jobId) + "/dataset/upload", {
            method: "POST",
            body: formData,
          })
            .then(function (r) {
              return r.json().then(function (up) {
                if (!r.ok) throw new Error(up.error || r.statusText);
                return up;
              });
            })
            .then(function (up) {
              showCreateSuccess(msgEl, d, up);
            });
        })
        .then(undefined, function (err) {
          msgEl.className = "message error";
          msgEl.textContent = "Error: " + (typeof err === "string" ? err : String(err));
        });
    });
  }

  function showCreateSuccess(msgEl, d, uploadResult) {
    msgEl.className = "message success";
    const started = d.status === "started";
    let html =
      "Experiment folder created. " +
      (started ? "Training started. " : "") +
      "Job ID: <strong>" +
      escapeHtml(d.job_id) +
      "</strong>. <a href=\"#/experiment/" +
      encodeURIComponent(d.job_id) +
      '">View experiment →</a>';
    if (uploadResult && uploadResult.uploaded && uploadResult.uploaded.length > 0) {
      html += "<p class=\"upload-summary\">Uploaded " + uploadResult.uploaded.length + " image(s) to dataset.</p>";
      if (uploadResult.errors && uploadResult.errors.length > 0) {
        html += "<p class=\"upload-errors\">Some files could not be uploaded: " + escapeHtml(uploadResult.errors.join("; ")) + "</p>";
      }
    }
    if (d.command) {
      html += '<p class="command-label">finetuning.py command:</p><pre class="command-block">' +
        escapeHtml(d.command) + "</pre>";
    }
    msgEl.innerHTML = html;
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function render() {
    const route = parseHash();
    if (route.view === "list") renderList();
    else if (route.view === "detail") renderDetail(route.name);
    else if (route.view === "new") renderNew();
  }

  window.addEventListener("hashchange", render);
  render();
})();
