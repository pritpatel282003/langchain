from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

# Example large text
text = """
# Base class (Parent)
class Animal:
    def __init__(self, name):
        self.name = name   # public attribute
        self._age = 0      # protected attribute (by convention)
        self.__id = 101    # private attribute (name mangling)

    def speak(self):
        return f" makes a sound."

    def get_id(self):  # getter for private attribute
        return self.__id


# Derived class (Child - Inheritance)
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)   # call parent constructor
        self.breed = breed

    # Polymorphism (method overriding)
    def speak(self):
        return f" barks."


class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name)
        self.color = color

    def speak(self):  # overriding
        return f" meows."


# Usage
dog1 = Dog("Tommy", "Labrador")
cat1 = Cat("Kitty", "White")

print(dog1.speak())       # Tommy barks.
print(cat1.speak())       # Kitty meows.
print(dog1.get_id())      # 101 (private accessed via getter)
print(dog1._age)          # 0 (protected, but accessible)
print(dog1.breed)         # Labrador

"""

# Initialize RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,   # max size of each chunk
    chunk_size=400,
    chunk_overlap=10
)

# Split text into chunks
chunks = text_splitter.split_text(text)

# Print results
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")
