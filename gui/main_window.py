"""Главное окно приложения: меню (загрузка/сохранение данных), вкладки
«Топология», «Сценарии», «Задача», «Алгоритм», «Эксперимент», «Результаты»
и строка статуса."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QTabWidget

from core import io
from core.logging_setup import get_logger
from examples.generate_examples import build_large, build_medium, build_small

from .state import AppState
from .tab_algorithm import AlgorithmTab
from .tab_experiment import ExperimentTab
from .tab_problem import ProblemTab
from .tab_results import ResultsTab
from .tab_scenarios import ScenariosTab
from .tab_topology import TopologyTab

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class MainWindow(QMainWindow):
    """Главное окно: держит общее состояние (AppState), шесть вкладок,
    меню «Файл»/«Справка» и строку статуса; синхронизирует вкладки между
    собой при смене данных или постановки задачи."""

    def __init__(self, parent=None):
        """Собрать интерфейс, загрузить пример medium.json по умолчанию
        и предложить бюджет (половина стоимости кандидатов)."""
        super().__init__(parent)
        self.log = get_logger("gui.main")
        self.state = AppState()
        self._load_default_example()

        self.setWindowTitle("OptimNet — оптимизация живучести компьютерной сети")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.topology_tab = TopologyTab(self.state)
        self.scenarios_tab = ScenariosTab(self.state)
        self.problem_tab = ProblemTab(self.state)
        self.algorithm_tab = AlgorithmTab(self.state)
        self.experiment_tab = ExperimentTab(self.state)
        self.results_tab = ResultsTab(self.state)

        self.tabs.addTab(self.topology_tab, "Топология")
        self.tabs.addTab(self.scenarios_tab, "Сценарии")
        self.tabs.addTab(self.problem_tab, "Задача")
        self.tabs.addTab(self.algorithm_tab, "Алгоритм")
        self.tabs.addTab(self.experiment_tab, "Эксперимент")
        self.tabs.addTab(self.results_tab, "Результаты")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self._build_menu()
        self.status = QLabel("")
        self.statusBar().addPermanentWidget(self.status)
        self._update_status()

        # синхронизация: смена постановки/режима меняет список алгоритмов
        self.problem_tab.formulation.currentIndexChanged.connect(self._on_config_changed)
        self.problem_tab.var_mode.currentIndexChanged.connect(self._on_config_changed)

        self.tabs.currentChanged.connect(lambda _: self.results_tab.refresh())
        self.log.info("Главное окно создано")

    # ------------------------------------------------------------- #

    def _load_default_example(self) -> None:
        """Поднять examples/medium.json в состояние при старте; при ошибке —
        приложение всё равно открывается с пустыми данными."""
        try:
            net, scenarios = io.load_example(EXAMPLES_DIR / "medium.json")
            self.state.network = net
            self.state.scenarios = scenarios
            self._suggest_budget()
        except Exception as exc:  # noqa: BLE001 — старт и без примера
            self.log.warning("Не удалось загрузить пример medium.json: %s", exc)

    @staticmethod
    def _suggest_budget_ratio() -> float:
        """Доля суммарной стоимости кандидатов, предлагаемая как бюджет."""
        return 0.5

    def _suggest_budget(self) -> None:
        """Разумный бюджет по умолчанию: половина стоимости всех кандидатов,
        чтобы ограничение 5.1 было существенным."""
        total = sum(c.cost for c in self.state.network.candidates)
        if total > 0:
            self.state.config.budget = round(total * self._suggest_budget_ratio(), 1)

    def _build_menu(self) -> None:
        """Меню «Файл» (открыть/сохранить пример, импорт/экспорт CSV,
        демо-примеры, выход) и «Справка» (о программе)."""
        m = self.menuBar().addMenu("Файл")

        open_act = QAction("Открыть пример (JSON)…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._open_example)
        m.addAction(open_act)

        save_act = QAction("Сохранить пример (JSON)…", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._save_example)
        m.addAction(save_act)

        m.addSeparator()
        import_csv = QAction("Импорт топологии (CSV-каталог)…", self)
        import_csv.triggered.connect(self._import_csv)
        m.addAction(import_csv)
        export_csv = QAction("Экспорт топологии (CSV-каталог)…", self)
        export_csv.triggered.connect(self._export_csv)
        m.addAction(export_csv)

        m.addSeparator()
        demo_menu = m.addMenu("Демо-примеры")
        for builder, name in [(build_small, "Малый (6 узлов)"),
                              (build_medium, "Средний (10 узлов)"),
                              (build_large, "Большой (20 узлов)")]:
            act = QAction(name, self)
            act.triggered.connect(lambda _=False, b=builder: self._load_demo(b))
            demo_menu.addAction(act)

        m.addSeparator()
        quit_act = QAction("Выход", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        m.addAction(quit_act)

        help_menu = self.menuBar().addMenu("Справка")
        about = QAction("О программе", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    # --------------------- действия файлов --------------------- #

    def _open_example(self) -> None:
        """Диалог открытия примера (JSON): загрузить сеть + сценарии,
        сбросить результаты, обновить все вкладки."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть пример", str(EXAMPLES_DIR), "JSON (*.json)")
        if not path:
            return
        try:
            net, scenarios = io.load_example(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            return
        self.state.network = net
        self.state.scenarios = scenarios
        self.state.last_results = {}
        self._suggest_budget()
        self._refresh_all()
        self.log.info("Загружен пример: %s", path)

    def _save_example(self) -> None:
        """Диалог сохранения текущих сети и сценариев в один JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить пример", "example.json", "JSON (*.json)")
        if path:
            io.save_example(self.state.network, self.state.scenarios, path)

    def _import_csv(self) -> None:
        """Импорт топологии из каталога с CSV-таблицами (nodes/links/…)."""
        path = QFileDialog.getExistingDirectory(self, "Каталог с CSV (nodes/links/…)")
        if not path:
            return
        try:
            self.state.network = io.load_network_csv(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка импорта", str(exc))
            return
        self.state.last_results = {}
        self._refresh_all()

    def _export_csv(self) -> None:
        """Экспорт текущей топологии в CSV-каталог."""
        path = QFileDialog.getExistingDirectory(self, "Каталог для CSV")
        if path:
            io.save_network_csv(self.state.network, path)

    def _load_demo(self, builder) -> None:
        """Поднять встроенный демо-пример (builder — build_small/medium/large),
        сбросить результаты и обновить вкладки."""
        net, scenarios = builder()
        self.state.network = net
        self.state.scenarios = scenarios
        self.state.last_results = {}
        self._suggest_budget()
        self._refresh_all()

    def _about(self) -> None:
        """Диалог «О программе»."""
        QMessageBox.about(
            self, "О программе",
            "OptimNet — ПО оптимизации живучести компьютерной сети.\n\n"
            "Сеть моделируется взвешенным ориентированным графом G = ⟨V, E, C, W⟩; "
            "живучесть Ψ(x) = Σ p(s)·φ(Q(x,s)) (doc/optim.md). Постановки 5.1–5.3 "
            "и аддитивная свёртка критериев; алгоритмы: генетический, имитация "
            "отжига, ветви и границы, жадная эвристика.\n\n"
            "Версия 1.0, Python + PyQt6.",
        )

    # --------------------- синхронизация вкладок --------------------- #

    def _on_tab_changed(self, index: int) -> None:
        """При смене вкладки — обновить её содержимое (метод refresh, если
        есть) и строку статуса."""
        widget = self.tabs.widget(index)
        if hasattr(widget, "refresh"):
            widget.refresh()
        self._update_status()

    def _on_config_changed(self) -> None:
        """Смена постановки/режима переменных — перестроить списки алгоритмов
        на вкладках «Алгоритм» и «Эксперимент»."""
        self.algorithm_tab.refresh()
        self.experiment_tab.refresh()

    def _refresh_all(self) -> None:
        """Обновить все вкладки и строку статуса (после загрузки данных)."""
        self.topology_tab.refresh()
        self.scenarios_tab.refresh()
        self.problem_tab.refresh()
        self.algorithm_tab.refresh()
        self.experiment_tab.refresh()
        self.results_tab.refresh()
        self._update_status()

    def _update_status(self) -> None:
        """Строка статуса: счётчики узлов/каналов/кандидатов/сценариев
        и контрольная сумма Σp(s)."""
        net, sc = self.state.network, self.state.scenarios
        total_p = sum(s.p for s in sc)
        self.status.setText(
            f"узлов: {len(net.nodes)}  |  каналов: {len(net.links)}  |  "
            f"кандидатов: {len(net.candidates)}  |  сценариев: {len(sc)}  |  "
            f"Σp(s) = {total_p:.3f}"
        )
