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
    if (parts[0] === "compare" && parts[1] && parts[2]) {
      return { view: "compare", exp1: parts[1], exp2: parts[2] };
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
          "</select>" +
          ' <button type="button" class="btn btn-small" id="compareBtn" disabled>Compare (0 selected)</button>';
        const table = document.createElement("table");
        table.className = "exp-table";
        table.innerHTML =
          "<thead><tr>" +
          "<th class=\"exp-table-col-select\"></th>" +
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
          row.dataset.expName = exp.name;
          row.innerHTML =
            '<td class="exp-table-col-select"><input type="checkbox" class="exp-select-cb" value="' + escapeHtml(exp.name) + '"></td>' +
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

        function updateCompareButton() {
          var checked = getMain().querySelectorAll(".exp-select-cb:checked");
          var btn = document.getElementById("compareBtn");
          if (!btn) return;
          var n = checked.length;
          btn.disabled = n !== 2;
          btn.textContent = n === 2 ? "Compare" : "Compare (" + n + " selected)";
        }
        getMain().querySelectorAll(".exp-select-cb").forEach(function (cb) {
          cb.addEventListener("change", function () {
            var checked = getMain().querySelectorAll(".exp-select-cb:checked");
            if (checked.length > 2) this.checked = false;
            updateCompareButton();
          });
        });
        var compareBtn = document.getElementById("compareBtn");
        if (compareBtn) {
          compareBtn.addEventListener("click", function () {
            var checked = getMain().querySelectorAll(".exp-select-cb:checked");
            if (checked.length !== 2) return;
            var names = [checked[0].value, checked[1].value].sort();
            window.location.hash = "#/compare/" + encodeURIComponent(names[0]) + "/" + encodeURIComponent(names[1]);
          });
        }
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
      '<button type="button" class="tab-btn" data-tab="compare_gen_dataset">Compare</button>' +
      '<button type="button" class="tab-btn" data-tab="finetune">Finetune</button>' +
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
      '<button class="btn" id="refreshLog">Refresh</button> ' +
      '<button class="btn" id="parseLogBtn" title="Parse raw log and regenerate metrics.json">Parse log</button>' +
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
      '<div class="generated-img-wrapper">' +
      '<div class="generated-img-toolbar"><label class="generated-compact-label"><input type="checkbox" id="generatedCompactView"> Compact view (no labels, no gap)</label></div>' +
      '<div class="img-grid dataset-grid" id="generatedImgGrid"></div><div id="generatedImgLoadMoreWrap" class="generated-load-more-wrap"></div></div>';

    const compareGenDatasetPanel = document.createElement("div");
    compareGenDatasetPanel.className = "tab-panel";
    compareGenDatasetPanel.setAttribute("data-tab", "compare_gen_dataset");
    compareGenDatasetPanel.innerHTML =
      '<div class="compare-gen-dataset-headers">' +
      '<span class="compare-gen-dataset-label">Dataset</span>' +
      '<span class="compare-gen-dataset-label">AI</span>' +
      '</div>' +
      '<div class="compare-gen-dataset-grid" id="compareGenDatasetGrid"></div>' +
      '<div id="compareGenDatasetLoadMore" class="generated-load-more-wrap"></div>';

    const finetunePanel = document.createElement("div");
    finetunePanel.className = "tab-panel";
    finetunePanel.setAttribute("data-tab", "finetune");
    finetunePanel.innerHTML =
      '<div class="finetune-command-section generate-form">' +
      '<p class="finetune-description">Run finetuning for this experiment. Dataset path points to this experiment\'s uploaded images.</p>' +
      '<div class="finetune-options">' +
      '<div class="form-row"><label for="finetuneMaxIter">max_iter (optional)</label><input type="number" id="finetuneMaxIter" placeholder="e.g. 5000" min="1"></div>' +
      '<div class="form-row"><label for="finetuneEpochs">epochs (optional)</label><input type="number" id="finetuneEpochs" placeholder="e.g. 50" min="1"></div>' +
      '</div>' +
      '<div class="form-row"><label class="gen-command-label">Command <button type="button" class="btn btn-small gen-copy-btn" id="finetuneCopyBtn">Copy</button></label><pre id="finetuneCommand" class="gen-command finetune-command"></pre></div>' +
      '<p class="finetune-hint">See <a href="/fine_tuning.md" target="_blank">fine_tuning.md</a> for full documentation.</p>' +
      '</div>';

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
    getMain().appendChild(compareGenDatasetPanel);
    getMain().appendChild(finetunePanel);
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
        if (tab === "generated_img") {
          var generatedCompactCb = getMain().querySelector("#generatedCompactView");
          if (generatedCompactCb) {
            var grid = document.getElementById("generatedImgGrid");
            if (grid) grid.classList.toggle("compact-view", generatedCompactCb.checked);
          }
        }
        if (tab === "compare_gen_dataset" && typeof loadCompareGenDataset === "function") {
          loadCompareGenDataset();
        }
        if (tab === "finetune" && typeof updateFinetuneCommand === "function") {
          updateFinetuneCommand();
        }
      });
    });

    (function () {
      var generatedCompactCb = getMain().querySelector("#generatedCompactView");
      if (generatedCompactCb) {
        generatedCompactCb.addEventListener("change", function () {
          var grid = document.getElementById("generatedImgGrid");
          if (grid) grid.classList.toggle("compact-view", this.checked);
        });
      }
    })();

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

    function updateFinetuneCommand() {
      var pre = document.getElementById("finetuneCommand");
      var maxIterEl = document.getElementById("finetuneMaxIter");
      var epochsEl = document.getElementById("finetuneEpochs");
      if (!pre) return;
      var maxIter = maxIterEl ? (maxIterEl.value || "").trim() : "";
      var epochs = epochsEl ? (epochsEl.value || "").trim() : "";
      var params = [];
      if (maxIter) params.push("max_iter=" + encodeURIComponent(maxIter));
      if (epochs) params.push("epochs=" + encodeURIComponent(epochs));
      var q = params.length ? "?" + params.join("&") : "";
      fetch(API + "/experiments/" + encodeURIComponent(name) + "/finetune-command" + q)
        .then(function (r) { return r.json(); })
        .then(function (d) {
          pre.textContent = d.command || "(error)";
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

    const INITIAL_COMPARE_COUNT = 200;
    const LOAD_MORE_COMPARE_COUNT = 200;

    function getCompareKey(fname) {
      if (!fname || typeof fname !== "string") return "";
      var base = fname.replace(/\.(png|jpg|jpeg)$/i, "");
      return base.replace(/^\d+_/, "");
    }

    function loadCompareGenDataset() {
      var grid = document.getElementById("compareGenDatasetGrid");
      var loadMoreWrap = document.getElementById("compareGenDatasetLoadMore");
      if (!grid) return;
      grid.textContent = "Loading…";
      Promise.all([
        fetch(API + "/experiments/" + encodeURIComponent(name) + "/dataset").then(function (r) { return r.ok ? r.json() : []; }),
        fetch(API + "/experiments/" + encodeURIComponent(name) + "/generated_img").then(function (r) { return r.ok ? r.json() : []; })
      ]).then(function (results) {
        var datasetList = Array.isArray(results[0]) ? results[0] : [];
        var genList = Array.isArray(results[1]) ? results[1] : [];
        var datasetByKey = {};
        datasetList.forEach(function (f) { if (f && f.name) { var k = getCompareKey(f.name); if (k) datasetByKey[k] = f.name; } });
        var genByKey = {};
        genList.forEach(function (f) { if (f && f.name) { var k = getCompareKey(f.name); if (k) genByKey[k] = f.name; } });
        var intersection = Object.keys(datasetByKey).filter(function (k) { return genByKey[k]; }).sort();
        grid.textContent = "";
        if (intersection.length === 0) {
          grid.innerHTML = "<p class=\"metrics-empty\">No overlap. Match by character key (e.g. 五.png and 022_五.png both key to 五).</p>";
          if (loadMoreWrap) loadMoreWrap.innerHTML = "";
          return;
        }
        var displayedCount = 0;

        function createCell(isDataset, fname) {
          var cell = document.createElement("div");
          cell.className = "compare-gen-dataset-cell";
          var label = document.createElement("span");
          label.className = "compare-gen-dataset-cell-label";
          label.textContent = isDataset ? "Dataset" : "AI";
          cell.appendChild(label);
          var a = document.createElement("a");
          var path = isDataset ? "dataset" : "generated_img";
          a.href = API + "/experiments/" + encodeURIComponent(name) + "/" + path + "/" + encodeURIComponent(fname);
          a.target = "_blank";
          var img = document.createElement("img");
          img.src = a.href;
          img.alt = fname;
          a.appendChild(img);
          cell.appendChild(a);
          return cell;
        }

        function createPairCard(key) {
          var datasetFname = datasetByKey[key];
          var genFname = genByKey[key];
          var card = document.createElement("div");
          card.className = "compare-gen-dataset-pair";
          var imgs = document.createElement("div");
          imgs.className = "compare-gen-dataset-pair-imgs";
          imgs.appendChild(createCell(true, datasetFname));
          imgs.appendChild(createCell(false, genFname));
          card.appendChild(imgs);
          var caption = document.createElement("span");
          caption.className = "img-caption";
          caption.textContent = key;
          card.appendChild(caption);
          return card;
        }

        function appendPairs(startIdx, count) {
          var endIdx = Math.min(startIdx + count, intersection.length);
          for (var i = startIdx; i < endIdx; i++) {
            grid.appendChild(createPairCard(intersection[i]));
          }
          displayedCount = endIdx;
        }

        function updateLoadMoreBtn() {
          if (!loadMoreWrap) return;
          if (displayedCount >= intersection.length) {
            loadMoreWrap.innerHTML = "";
            return;
          }
          var remaining = intersection.length - displayedCount;
          var loadCount = Math.min(LOAD_MORE_COMPARE_COUNT, remaining);
          loadMoreWrap.innerHTML = '<button type="button" class="btn btn-load-more">Load more (' + loadCount + ' of ' + remaining + ' remaining)</button>';
          loadMoreWrap.querySelector("button").addEventListener("click", function () {
            appendPairs(displayedCount, LOAD_MORE_COMPARE_COUNT);
            updateLoadMoreBtn();
          });
        }

        appendPairs(0, INITIAL_COMPARE_COUNT);
        updateLoadMoreBtn();
      }).catch(function () {
        grid.textContent = "Failed to load.";
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
    loadCompareGenDataset();

    const refreshBtn = document.getElementById("refreshLog");
    if (refreshBtn) refreshBtn.addEventListener("click", function () { loadLog(); loadMetrics(); });

    const parseLogBtn = document.getElementById("parseLogBtn");
    if (parseLogBtn) {
      parseLogBtn.addEventListener("click", function () {
        parseLogBtn.disabled = true;
        parseLogBtn.textContent = "Parsing…";
        var base = API + "/experiments/" + encodeURIComponent(name);
        fetch(base + "/parse-log", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
          .then(function (r) {
            if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || r.statusText); });
            return r.json();
          })
          .then(function (d) {
            parseLogBtn.textContent = "Parse log";
            parseLogBtn.disabled = false;
            loadMetrics();
          })
          .catch(function (err) {
            parseLogBtn.textContent = "Parse log";
            parseLogBtn.disabled = false;
            alert("Parse failed: " + (err.message || String(err)));
          });
      });
    }

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
    ["finetuneMaxIter", "finetuneEpochs"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.addEventListener("input", updateFinetuneCommand);
    });
    var finetuneCopyBtn = document.getElementById("finetuneCopyBtn");
    if (finetuneCopyBtn) {
      finetuneCopyBtn.addEventListener("click", function () {
        var pre = document.getElementById("finetuneCommand");
        var text = pre ? pre.textContent : "";
        if (!text || text.startsWith("(")) return;
        navigator.clipboard.writeText(text).then(function () {
          var orig = finetuneCopyBtn.textContent;
          finetuneCopyBtn.textContent = "Copied!";
          setTimeout(function () { finetuneCopyBtn.textContent = orig; }, 1500);
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
    else if (route.view === "compare") renderCompare(route.exp1, route.exp2);
    else if (route.view === "new") renderNew();
  }

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

  function renderCompare(exp1, exp2) {
    getMain().innerHTML = '<p class="loading">Loading…</p>';
    var back = '<a href="#/" class="back-link">← Back to list</a>';
    var title = '<p class="page-title">Compare: ' + escapeHtml(exp1) + ' vs ' + escapeHtml(exp2) + '</p>';
    var html =
      back + title +
      '<div class="compare-chart-controls">' +
      '<span class="chart-label">Chart columns:</span> <div id="compareChartSelect" class="chart-column-select"></div> ' +
      '<button type="button" class="btn btn-small" id="compareChartClear">Clear all</button>' +
      '</div>' +
      '<div class="compare-layout">' +
      '<div class="compare-col">' +
      '<h3 class="compare-col-title"><a href="#/experiment/' + encodeURIComponent(exp1) + '">' + escapeHtml(exp1) + '</a></h3>' +
      '<div class="compare-summary" id="compareSummary1"></div>' +
      '<div class="chart-section"><div class="chart-wrap"><canvas id="compareChart1"></canvas></div></div>' +
      '<div id="compareTable1" class="compare-table-wrap"></div>' +
      '</div>' +
      '<div class="compare-col">' +
      '<h3 class="compare-col-title"><a href="#/experiment/' + encodeURIComponent(exp2) + '">' + escapeHtml(exp2) + '</a></h3>' +
      '<div class="compare-summary" id="compareSummary2"></div>' +
      '<div class="chart-section"><div class="chart-wrap"><canvas id="compareChart2"></canvas></div></div>' +
      '<div id="compareTable2" class="compare-table-wrap"></div>' +
      '</div></div>' +
      '<div class="compare-generated-section">' +
      '<h3 class="compare-section-title">Generated images</h3>' +
      '<div class="compare-gen-headers">' +
      '<span class="compare-gen-label">' + escapeHtml(exp1) + '</span>' +
      '<span class="compare-gen-label">' + escapeHtml(exp2) + '</span>' +
      '</div>' +
      '<div class="compare-gen-grid" id="compareGenImgGrid"></div>' +
      '<div id="compareGenImgLoadMore" class="generated-load-more-wrap"></div>' +
      '</div>';
    getMain().innerHTML = html;

    var chart1 = null, chart2 = null;
    var blocks1 = [], blocks2 = [];
    var chartDefaults = ["AC_acc_c", "AC_acc_s", "AC_g_acc_c", "AC_g_acc_s"];

    function fetchMetrics(name) {
      var base = API + "/experiments/" + encodeURIComponent(name);
      return fetch(base + "/metrics")
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
        });
    }

    function fetchExperiment(name) {
      return fetch(API + "/experiments?sort=time").then(function (r) { return r.json(); })
        .then(function (items) {
          var exp = items.find(function (e) { return e.name === name; });
          return exp || {};
        });
    }

    function renderChart(canvasId, blocks, selectedKeys) {
      var canvas = document.getElementById(canvasId);
      if (!canvas || typeof Chart === "undefined") return;
      if (!selectedKeys || selectedKeys.length === 0) {
        if (canvasId === "compareChart1" && chart1) { chart1.destroy(); chart1 = null; }
        if (canvasId === "compareChart2" && chart2) { chart2.destroy(); chart2 = null; }
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
      if (canvasId === "compareChart1" && chart1) chart1.destroy();
      if (canvasId === "compareChart2" && chart2) chart2.destroy();
      var ch = new Chart(canvas, {
        type: "line",
        data: { labels: steps, datasets: datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { intersect: false, mode: "index" },
          onHover: function (evt, elements, chart) {
            var other = chart === chart1 ? chart2 : chart1;
            if (!other) return;
            if (!elements || elements.length === 0) {
              other.setActiveElements([]);
              other.tooltip.setActiveElements([], { x: 0, y: 0 });
            } else {
              var idx = elements[0].index;
              var n = other.data.datasets.length;
              var active = [];
              for (var d = 0; d < n; d++) {
                var maxIdx = other.data.datasets[d].data.length - 1;
                active.push({ datasetIndex: d, index: Math.min(idx, maxIdx) });
              }
              other.setActiveElements(active);
              var meta = other.getDatasetMeta(0);
              if (meta && meta.data && meta.data.length > 0) {
                var el = meta.data[Math.min(idx, meta.data.length - 1)];
                other.tooltip.setActiveElements(active, { x: el.x, y: el.y });
              }
            }
            other.update("none");
          },
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
                title: function (items) { return "Step " + (items[0] && items[0].label != null ? items[0].label : ""); },
                label: function (ctx) {
                  var v = ctx.parsed.y;
                  var s = v != null && !isNaN(v) ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : "-";
                  return ctx.dataset.label + ": " + s;
                },
              },
            },
          },
          scales: { x: { title: { display: true, text: "Step" } }, y: { beginAtZero: false } },
        },
      });
      if (canvasId === "compareChart1") chart1 = ch; else chart2 = ch;
    }

    function updateBothCharts(selectedKeys) {
      renderChart("compareChart1", blocks1, selectedKeys);
      renderChart("compareChart2", blocks2, selectedKeys);
    }

    function buildKeys(blocks) {
      var keys = [], seen = {};
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
      return keys;
    }

    function renderMetricsTable(blocks, containerId, expName) {
      var container = document.getElementById(containerId);
      if (!container) return;
      if (!blocks || blocks.length === 0) {
        container.innerHTML = "<p class=\"metrics-empty\">No metrics.</p>";
        return;
      }
      var keys = buildKeys(blocks);
      var html = "<table class=\"metrics-table\"><thead><tr>";
      keys.forEach(function (k) { html += "<th>" + escapeHtml(k) + "</th>"; });
      html += "</tr></thead><tbody>";
      blocks.slice(-20).forEach(function (row) {
        html += "<tr>";
        keys.forEach(function (k) {
          var v = row[k];
          html += "<td>" + escapeHtml(v === undefined || v === null ? "" : String(v)) + "</td>";
        });
        html += "</tr>";
      });
      html += "</tbody></table>";
      if (blocks.length > 20 && expName) {
        html += "<p class=\"compare-table-note\">Last 20 rows. <a href=\"#/experiment/" + encodeURIComponent(expName) + "\">View full table</a></p>";
      }
      container.innerHTML = html;
    }

    function setSummary(elId, exp) {
      var el = document.getElementById(elId);
      if (!el) return;
      if (!exp || !exp.name) {
        el.textContent = "—";
        return;
      }
      var parts = [];
      if (exp.n_checkpoints != null) parts.push(exp.n_checkpoints + " checkpoints");
      if (exp.dataset_count != null && exp.dataset_count > 0) parts.push(exp.dataset_count + " dataset");
      if (exp.n_images != null) parts.push(exp.n_images + " images");
      el.textContent = parts.length ? parts.join(" · ") : "—";
    }

    function fetchGeneratedImg(name) {
      return fetch(API + "/experiments/" + encodeURIComponent(name) + "/generated_img")
        .then(function (r) { return r.ok ? r.json() : []; });
    }

    var INITIAL_GEN_COUNT = 200;
    var LOAD_MORE_GEN_COUNT = 200;

    function renderGeneratedImgCompare(list1, list2) {
      var grid = document.getElementById("compareGenImgGrid");
      var loadMoreWrap = document.getElementById("compareGenImgLoadMore");
      if (!grid) return;
      var names1 = {};
      list1.forEach(function (f) { if (f && f.name) names1[f.name] = true; });
      var names2 = {};
      list2.forEach(function (f) { if (f && f.name) names2[f.name] = true; });
      var allFilenames = Object.keys(names1).concat(Object.keys(names2)).filter(function (n, i, arr) { return arr.indexOf(n) === i; }).sort();
      if (allFilenames.length === 0) {
        grid.innerHTML = "<p class=\"metrics-empty\">No generated images in either experiment.</p>";
        if (loadMoreWrap) loadMoreWrap.innerHTML = "";
        return;
      }
      var displayedCount = 0;

      function createImgCell(expName, fname, hasImage) {
        var cell = document.createElement("div");
        cell.className = "compare-gen-cell";
        if (hasImage) {
          var a = document.createElement("a");
          a.href = API + "/experiments/" + encodeURIComponent(expName) + "/generated_img/" + encodeURIComponent(fname);
          a.target = "_blank";
          var img = document.createElement("img");
          img.src = a.href;
          img.alt = fname;
          a.appendChild(img);
          cell.appendChild(a);
        } else {
          var placeholder = document.createElement("div");
          placeholder.className = "compare-gen-placeholder";
          placeholder.textContent = "—";
          cell.appendChild(placeholder);
        }
        return cell;
      }

      function createPairCard(fname) {
        var card = document.createElement("div");
        card.className = "compare-gen-pair";
        var imgs = document.createElement("div");
        imgs.className = "compare-gen-pair-imgs";
        imgs.appendChild(createImgCell(exp1, fname, names1[fname]));
        imgs.appendChild(createImgCell(exp2, fname, names2[fname]));
        card.appendChild(imgs);
        var caption = document.createElement("span");
        caption.className = "img-caption";
        caption.textContent = fname;
        card.appendChild(caption);
        return card;
      }

      function appendGenImages(startIdx, count) {
        var endIdx = Math.min(startIdx + count, allFilenames.length);
        for (var i = startIdx; i < endIdx; i++) {
          grid.appendChild(createPairCard(allFilenames[i]));
        }
        displayedCount = endIdx;
      }

      function updateLoadMoreBtn() {
        if (!loadMoreWrap) return;
        if (displayedCount >= allFilenames.length) {
          loadMoreWrap.innerHTML = "";
          return;
        }
        var remaining = allFilenames.length - displayedCount;
        var loadCount = Math.min(LOAD_MORE_GEN_COUNT, remaining);
        loadMoreWrap.innerHTML = '<button type="button" class="btn btn-load-more">Load more (' + loadCount + ' of ' + remaining + ' remaining)</button>';
        loadMoreWrap.querySelector("button").addEventListener("click", function () {
          appendGenImages(displayedCount, LOAD_MORE_GEN_COUNT);
          updateLoadMoreBtn();
        });
      }

      appendGenImages(0, INITIAL_GEN_COUNT);
      updateLoadMoreBtn();
    }

    Promise.all([
      fetchMetrics(exp1),
      fetchMetrics(exp2),
      fetchExperiment(exp1),
      fetchExperiment(exp2),
      fetchGeneratedImg(exp1),
      fetchGeneratedImg(exp2),
    ]).then(function (results) {
      blocks1 = Array.isArray(results[0]) ? results[0] : [];
      blocks2 = Array.isArray(results[1]) ? results[1] : [];
      var expData1 = results[2] || {};
      var expData2 = results[3] || {};
      var genList1 = Array.isArray(results[4]) ? results[4] : [];
      var genList2 = Array.isArray(results[5]) ? results[5] : [];
      setSummary("compareSummary1", expData1);
      setSummary("compareSummary2", expData2);
      renderMetricsTable(blocks1, "compareTable1", exp1);
      renderMetricsTable(blocks2, "compareTable2", exp2);
      renderGeneratedImgCompare(genList1, genList2);

      var allKeys = buildKeys(blocks1.concat(blocks2));
      var chartableKeys = allKeys.filter(function (k) { return isChartable(blocks1, k) || isChartable(blocks2, k); });
      var selectWrap = document.getElementById("compareChartSelect");
      if (selectWrap) {
        selectWrap.innerHTML = "";
        chartableKeys.forEach(function (k) {
          var label = document.createElement("label");
          label.className = "chart-check";
          var cb = document.createElement("input");
          cb.type = "checkbox";
          cb.dataset.key = k;
          if (chartDefaults.indexOf(k) >= 0) cb.checked = true;
          cb.addEventListener("change", function () {
            var sel = Array.from(selectWrap.querySelectorAll("input:checked")).map(function (x) { return x.dataset.key; });
            updateBothCharts(sel);
          });
          label.appendChild(cb);
          label.appendChild(document.createTextNode(" " + k));
          selectWrap.appendChild(label);
        });
        var sel = Array.from(selectWrap.querySelectorAll("input:checked")).map(function (x) { return x.dataset.key; });
        updateBothCharts(sel);
      }
      var clearBtn = document.getElementById("compareChartClear");
      if (clearBtn) {
        clearBtn.onclick = function () {
          selectWrap.querySelectorAll("input[type=checkbox]").forEach(function (cb) { cb.checked = false; });
          updateBothCharts([]);
        };
      }
    }).catch(function (err) {
      getMain().innerHTML = '<p class="message error">Failed to load: ' + escapeHtml(err.message || "Unknown error") + "</p>";
    });
  }

  window.addEventListener("hashchange", render);
  render();
})();
