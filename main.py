import numpy
import random

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock

from data import load_data
from clustering import matrix_initialization
from recommandation import recommend_top_videos
from matrix_factorization import matrix_factorization


NUM_VIDEOS_TO_RATE = 8


class VideoRecommendationApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.R = None
        self.themes = None
        self.titles = None
        self.user_ratings = None
        self.video_ratings = []
        self.status_label = None
        self.output_label = None

    def build(self):
        self.load_data()
        self.setup_layout()
        return self.layout

    def load_data(self):
        self.R, self.themes, self.titles = load_data('data.csv')
        self.R = numpy.array(self.R)
        self.N = len(self.R)
        self.M = len(self.R[0])
        self.K = len(set(self.themes))

    def setup_layout(self):
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.add_title_label()
        self.select_videos()
        self.add_submit_button()
        self.add_status_label()
        self.add_output_label()

    def add_title_label(self):
        title_label = Label(text="Veuillez noter chaque vidéo de 1 à 5", size_hint=(
            1, None), height=50, font_size=24, bold=True)
        self.layout.add_widget(title_label)

    def select_videos(self):
        theme_videos = {}
        for i, theme in enumerate(self.themes):
            if theme not in theme_videos:
                theme_videos[theme] = []
            theme_videos[theme].append(i)

        selected_videos = []
        for theme, videos in theme_videos.items():
            if videos:
                selected_video = random.choice(videos)
                selected_videos.append(selected_video)

        remaining_videos_count = NUM_VIDEOS_TO_RATE - len(selected_videos)
        if remaining_videos_count > 0:
            available_videos = set(range(self.M)) - set(selected_videos)
            additional_videos = random.sample(
                sorted(available_videos), remaining_videos_count)
            selected_videos.extend(additional_videos)

        for i in selected_videos:
            video_layout = BoxLayout(orientation='horizontal', spacing=5)
            title_label = Label(text=self.titles[i], size_hint=(
                0.7, None), height=30, font_size=16)
            theme_label = Label(text=f"({self.themes[i]})", size_hint=(
                0.3, None), height=30, font_size=16)
            rating_input = TextInput(
                hint_text="Votre note (1-5)", size_hint=(0.3, None), height=30)
            video_layout.add_widget(title_label)
            video_layout.add_widget(theme_label)
            video_layout.add_widget(rating_input)
            self.video_ratings.append((i, rating_input))
            self.layout.add_widget(video_layout)

    def add_submit_button(self):
        submit_button = Button(text="Soumettre les notes",
                               size_hint=(1, None), height=40, background_color=(0.2, 0.6, 1, 1))
        submit_button.bind(on_press=self.submit_ratings)
        self.layout.add_widget(submit_button)

    def add_status_label(self):
        self.status_label = Label(size_hint=(1, None), height=30, font_size=16)
        self.layout.add_widget(self.status_label)

    def add_output_label(self):
        self.output_label = Label(size_hint=(
            1, None), height=100, font_size=16)
        self.layout.add_widget(self.output_label)

    def submit_ratings(self, instance):
        user_ratings = [0] * self.M
        for i, rating_input in self.video_ratings:
            try:
                rating = int(rating_input.text)
                if rating < 1 or rating > 5:
                    raise ValueError(
                        "La note doit être comprise entre 1 et 5.")
                user_ratings[i] = rating
            except ValueError as e:
                self.status_label.text = f"Erreur : {e}"
                return
        self.user_ratings = user_ratings
        self.status_label.text = "Calcul en cours..."
        Clock.schedule_once(self.calculate_recommendations, 0.1)

    def calculate_recommendations(self, dt):
        if self.user_ratings:
            R = numpy.vstack([self.R, self.user_ratings])
            N = len(R)
            R = R / 5
            P, Q = matrix_initialization(R, N, self.M, self.K)
            nP, nQ = matrix_factorization(R, P, Q, self.K)
            nR = numpy.dot(nP, nQ.T)
            user_index = N - 1
            top_videos = recommend_top_videos(
                user_index, R, nR, self.titles, self.themes, n=5)
            self.output_label.text = "Recommendations:\n"
            for i, (title, theme, prediction) in enumerate(top_videos, start=1):
                self.output_label.text += f"{i}. {title} ({theme}) - {prediction}\n"
            self.status_label.text = ""


if __name__ == '__main__':
    VideoRecommendationApp().run()
