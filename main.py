import cv2
import argparse
import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional
from ultralytics import YOLO
from dataclasses import dataclass


class TableState(Enum):
    EMPTY = 'empty'
    OCCUPIED = 'occupied'
    APPROACHING = 'approaching'


@dataclass
class Event:
    timestamp: float
    state: TableState
    duration_sec: Optional[float] = None

class TableTracker:
    def __init__(self, empty_threshold_sec: float = 2.0, leave_threshold_sec: float = 2.0):
        self.state = TableState.EMPTY
        self.empty_since: Optional[float] = None # Время, когда стол стал пустым
        self.occupied_since: Optional[float] = None # Время, когда стол стал занятым
        
        # Добавляем фиксацию времени, когда мы в последний раз видели человека
        self.last_seen_person: Optional[float] = None 
        
        self.empty_threshold = empty_threshold_sec # Буфер для определения подхода к столу (например, 2 сек)
        self.leave_threshold = leave_threshold_sec  # Буфер от ложных исчезновений
        self.events: list[Event] = []
        self.max_timestamp = 0.0 # Для отслеживания общего времени наблюдения

    def update(self, timestamp: float, is_occupied: bool):
        if is_occupied:
            # Обновляем таймер: человек в кадре
            self.last_seen_person = timestamp 

            # Обновляем максимальный timestamp для корректного подсчёта общего времени наблюдения
            self.max_timestamp = max(self.max_timestamp, timestamp)
            
            # Если стол был пустым и вдруг появилось движение — это подход
            if self.state == TableState.EMPTY and self.empty_since is not None:
                time_empty = timestamp - self.empty_since
                if time_empty >= self.empty_threshold:
                    self.events.append(Event(timestamp=timestamp, state=TableState.APPROACHING, duration_sec=time_empty))
                    print(f"[{timestamp:.1f}] → Подход к столу (пустовал {time_empty:.1f} сек)")

            if self.state != TableState.OCCUPIED:
                self.state = TableState.OCCUPIED
                self.occupied_since = timestamp
                self.empty_since = None

        else:
            # Человека нет в кадре
            if self.state == TableState.OCCUPIED:
                # Считаем, сколько времени прошло с момента, когда мы последний раз видели человека
                time_since_last_seen = timestamp - (self.last_seen_person or timestamp)
                
                # Завершаем сессию, только если человека нет дольше, чем leave_threshold (например, 2 сек)
                if time_since_last_seen >= self.leave_threshold:
                    if self.occupied_since is not None:
                        # Длительность считаем до момента, когда человека видели в последний раз
                        duration = self.last_seen_person - self.occupied_since
                        
                        # ИСПРАВЛЕНИЕ: Теперь мы записываем событие как OCCUPIED
                        self.events.append(Event(timestamp=self.last_seen_person, state=TableState.OCCUPIED, duration_sec=duration))
                        print(f"[{timestamp:.1f}] → Стол пустой (был занят {duration:.1f} сек)")
                        
                    self.state = TableState.EMPTY
                    # Считаем, что стол пустует с момента последней детекции человека
                    self.empty_since = self.last_seen_person
                    self.occupied_since = None

    def get_statistics(self) -> pd.DataFrame: # Преобразуем события в DataFrame для анализа
        data = [{
            'timestamp': event.timestamp,
            'state': event.state.value,
            'duration_sec': event.duration_sec
        } for event in self.events]

        return pd.DataFrame(data)

    def get_analytics(self) -> dict: 
        """
        Подсчёт итоговой статистики с фильтрацией выбросов.

        Returns:
            dict: Словарь с метриками
        """
        if not self.events:
            return {'error': 'Нет событий для анализа'}

        df = self.get_statistics()

        # Фильтрация выбросов по длительности занятия стола
        # (исключаем слишком короткие < 1 сек и слишком длинные > 10 мин)
        occupied_events = df[
            (df['state'] == TableState.OCCUPIED.value) &
            (df['duration_sec'] >= 1.0) #&
            # (df['duration_sec'] <= 600.0)
        ]

        # Подход к столу (фильтруем ложные срабатывания < 0.5 сек)
        approach_events = df[
            (df['state'] == TableState.APPROACHING.value) &
            (df['duration_sec'] >= 0.5)
        ]

        # Метрики
        analytics = {
            'total_events': len(df),
            'occupied_count': len(occupied_events),
            'approach_count': len(approach_events),
        }

        # Средняя длительность занятия стола
        if len(occupied_events) > 0:
            analytics['avg_occupied_duration'] = occupied_events['duration_sec'].mean()
            analytics['min_occupied_duration'] = occupied_events['duration_sec'].min()
            analytics['max_occupied_duration'] = occupied_events['duration_sec'].max()
            analytics['median_occupied_duration'] = occupied_events['duration_sec'].median()
        else:
            analytics['avg_occupied_duration'] = 0
            analytics['min_occupied_duration'] = 0
            analytics['max_occupied_duration'] = 0
            analytics['median_occupied_duration'] = 0

        # Среднее время до подхода (после того как стол стал пустым)
        if len(approach_events) > 0:
            analytics['avg_approach_delay'] = approach_events['duration_sec'].mean()
            analytics['min_approach_delay'] = approach_events['duration_sec'].min()
            analytics['max_approach_delay'] = approach_events['duration_sec'].max()
        else:
            analytics['avg_approach_delay'] = 0
            analytics['min_approach_delay'] = 0
            analytics['max_approach_delay'] = 0

        # Общее время наблюдения
        analytics['total_observation_time'] = self.max_timestamp

        # Процент времени, когда стол был занят
        if analytics['total_observation_time'] > 0:
            total_occupied_time = occupied_events['duration_sec'].sum() if len(occupied_events) > 0 else 0
            analytics['occupancy_rate'] = total_occupied_time / analytics['total_observation_time'] * 100
        else:
            analytics['occupancy_rate'] = 0

        return analytics

    def finalize(self) -> None:
        """Принудительно закрывает активные сессии в конце видео."""
        if self.state == TableState.OCCUPIED and self.occupied_since is not None:
            end_time = self.last_seen_person or self.max_timestamp
            duration = end_time - self.occupied_since
            if duration >= 1.0: # Исключаем миллисекундные выбросы в конце
                self.events.append(Event(
                    timestamp=end_time, 
                    state=TableState.OCCUPIED, 
                    duration_sec=duration
                ))
                print(f"[{end_time:.1f}] → Стол пустой (Конец видео, был занят {duration:.1f} сек)")

    def print_analytics_report(self) -> None:
        analytics = self.get_analytics()
        if 'error' in analytics:
            print(analytics['error'])
            return

        print("\n=== Аналитический отчёт по занятости столика ===")
        print(f"Всего событий: {analytics['total_events']}")
        print(f"Количество занятых периодов: {analytics['occupied_count'] if analytics['occupied_count'] > 0 else 'Нет данных'}")
        print(f"Количество подходов к столу: {analytics['approach_count']}")
        print(f"Средняя длительность занятия стола: {analytics['avg_occupied_duration']:.1f} сек")
        print(f"Минимальная длительность занятия стола: {analytics['min_occupied_duration']:.1f} сек")
        print(f"Максимальная длительность занятия стола: {analytics['max_occupied_duration']:.1f} сек")
        print(f"Медианная длительность занятия стола: {analytics['median_occupied_duration']:.1f} сек")
        print(f"Среднее время до подхода к столу: {analytics['avg_approach_delay']:.1f} сек")
        print(f"Минимальное время до подхода к столу: {analytics['min_approach_delay']:.1f} сек")
        print(f"Максимальное время до подхода к столу: {analytics['max_approach_delay']:.1f} сек")
        print(f"Общее время наблюдения: {analytics['total_observation_time']:.1f} сек")
        print(f"Процент времени, когда стол был занят: {analytics['occupancy_rate']:.2f}%")

def setup_window(
        width: int,
        height: int,
        window_name: str = 'Select table',
        screen_width: int = 1920,
        screen_height: int = 1080
) -> None:
    """
    Настройка окна для отображения видео.
    Если видео больше, чем размер экрана, то оно
    будет масштабироваться, чтобы поместиться на экране.
    """
    scale = min(screen_width / width, screen_height / height)

    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, new_width, new_height)


def select_table_roi(video_path: str, window_name: str = 'Select table') -> tuple[int, int, int, int]:
    """
    Выбор области интереса (ROI) на видео.

    Returns:
        tuple: Координаты ROI в формате (x, y, width, height).

    Raises:
        ValueError: Если видео не может быть открыто или ROI не выбран.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        cap.release()
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    height, width = frame.shape[:2]
    setup_window(width, height, window_name=window_name)
    


    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        cap.release()
        cv2.destroyAllWindows()
        raise ValueError("Окно для выбора ROI было закрыто. Пожалуйста, выберите область интереса.")

    cap.release()
    cv2.destroyWindow(window_name)

    x, y, w, h = roi

    if w == 0 or h == 0:
        raise ValueError("ROI не выбран. Пожалуйста, выберите область интереса.")

    print(f"Выбранная область ROI: x={x}, y={y}, width={w}, height={h}")
    return roi


def detect_people_in_roi(frame: np.ndarray, roi: tuple[int, int, int, int], model: YOLO) -> tuple[bool, list]:
    """
    Детекция людей в ROI с помощью YOLO.

    Returns:
        tuple: (есть ли люди, список bbox людей в ROI)
    """
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]

    # Детекция людей (класс 0 = person в COCO)
    results = model(roi_frame, classes=[0], verbose=False)
    result = results[0]

    people_boxes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > 0.5:  # Порог уверенности
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2] в координатах ROI
                people_boxes.append(bbox)

    is_occupied = len(people_boxes) > 0
    return is_occupied, people_boxes


def background_subtraction(video_path: str, roi: tuple[int, int, int, int]) -> None:
    """
    Обработка видео с детекцией людей через YOLO.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Загружаем YOLOv8 nano (быстрая версия)
    print("Загрузка YOLO модели...")
    model = YOLO('yolov8n.pt')

    tracker = TableTracker(empty_threshold_sec=2.0)

    # Настройка окон до цикла обработки видео
    roi_width, roi_height = roi[2], roi[3]
    setup_window(roi_width, roi_height, window_name='ROI View')
    setup_window(roi_width, roi_height, window_name='Table State')
    cv2.namedWindow('Table State')

    # ОПТИМИЗАЦИЯ: Детектируем не каждый кадр
    DETECT_EVERY_N_FRAMES = 2  # Детекция каждые 2 кадра (экономия ~50%)

    frame_count = 0
    last_is_occupied = False
    last_people_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps  # Время в секундах

        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]

        # ОПТИМИЗАЦИЯ: Детекция только каждые N кадров
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            is_occupied, people_boxes = detect_people_in_roi(frame, roi, model)
            last_is_occupied = is_occupied
            last_people_boxes = people_boxes
        else:
            # Используем последние известные данные
            is_occupied = last_is_occupied
            people_boxes = last_people_boxes

        # Обновление состояния
        tracker.update(timestamp, is_occupied)

        # Визуализация: цвет рамки в зависимости от состояния
        state_colors = {
            TableState.EMPTY: (0, 255, 0),      # Зелёный
            TableState.OCCUPIED: (0, 0, 255),   # Красный
            TableState.APPROACHING: (0, 255, 255),  # Жёлтый
        }
        color = state_colors[tracker.state]

        # Рисуем рамку столика на основном кадре
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, f'State: {tracker.state.value}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Рисуем bbox людей внутри ROI
        for bbox in people_boxes:
            x1, y1, x2, y2 = map(int, bbox)
            # Координаты bbox в ROI → координаты в оригинальном кадре
            cv2.rectangle(roi_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Добавляем инфо о FPS для отладки
        cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Показываем окна
        cv2.imshow('ROI View', roi_frame)
        cv2.imshow('Table State', frame)

        # Проверка нажатия клавиши или закрытия окна
        key = cv2.waitKey(1) & 0xFF  # Минимальная задержка для скорости
        if key == ord('q'):
            break

        # Проверка: если окна закрыты — завершаем
        if cv2.getWindowProperty('ROI View', cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.getWindowProperty('Table State', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Финализируем трекер, чтобы закрыть активные сессии
    tracker.finalize()

    # Печатаем статистику после обработки видео
    tracker.print_analytics_report()

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Детекция занятости столика по видео'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Путь к видеофайлу'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    window_name = 'Select table'

    try:
        roi = select_table_roi(args.video, window_name)
        print(f"Координаты ROI: {roi}")
        background_subtraction(args.video, roi)

    except ValueError as e:
        print(e)
