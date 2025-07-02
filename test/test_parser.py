import dataclasses
import json
import parser
import sys
import tempfile
import unittest
from parser import auto_cli, get_all_parser, parse_all_args
from typing import Dict, List, Optional


class TestParser(unittest.TestCase):
    def test_string(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            name: str
            gender: str = 'male'
        parser = Args.get_parser()

        # Test specified value
        ns = parser.parse_args('--name Alice --gender female'.split())
        obj = Args.parse_namespace(ns)
        self.assertEqual(obj.name, 'Alice')
        self.assertEqual(obj.gender, 'female')

        # Test default value
        ns = parser.parse_args('--name Bob'.split())
        obj = Args.parse_namespace(ns)
        self.assertEqual(obj.name, 'Bob')
        self.assertEqual(obj.gender, 'male')

    def test_numbers(self):
        @auto_cli
        @dataclasses.dataclass
        class Point:
            x: int
            y: float = 0.0

        parser = Point.get_parser()
        # Test specified value
        parser.parse_args('--x 10 --y 20.5'.split())
        ns = parser.parse_args('--x 10 --y 20.5'.split())
        obj = Point.parse_namespace(ns)
        self.assertEqual(obj.x, 10)
        self.assertEqual(obj.y, 20.5)

        # Test default value
        ns = parser.parse_args('--x 5'.split())
        obj = Point.parse_namespace(ns)
        self.assertEqual(obj.x, 5)
        self.assertEqual(obj.y, 0.0)

    def test_boolean(self):
        @auto_cli
        @dataclasses.dataclass
        class Config:
            debug: bool = False
            verbose: bool = True

        parser = Config.get_parser()
        # Test specified value
        ns = parser.parse_args('--debug --no-verbose'.split())
        obj = Config.parse_namespace(ns)
        self.assertTrue(obj.debug)
        self.assertFalse(obj.verbose)

        # Test default value
        ns = parser.parse_args('--no-debug'.split())
        obj = Config.parse_namespace(ns)
        self.assertFalse(obj.debug)
        self.assertTrue(obj.verbose)

        ns = parser.parse_args('--verbose'.split())
        obj = Config.parse_namespace(ns)
        self.assertFalse(obj.debug)
        self.assertTrue(obj.verbose)

        ns = parser.parse_args([])
        obj = Config.parse_namespace(ns)
        self.assertFalse(obj.debug)
        self.assertTrue(obj.verbose)

    def test_list(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            items: List[int] = dataclasses.field(default_factory=list)

        parser = Args.get_parser()
        # Test specified value
        ns = parser.parse_args([
            '--items', '[1, 2, 3]'
        ])
        obj = Args.parse_namespace(ns)
        self.assertEqual(obj.items, [1, 2, 3])

        # Test default value
        ns = parser.parse_args([])
        obj = Args.parse_namespace(ns)
        self.assertEqual(obj.items, [])

    def test_dict(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            config: Dict[str, str] = dataclasses.field(default_factory=dict)

        parser = Args.get_parser()
        # Test specified value
        ns = parser.parse_args([
            '--config', '{"key1": "value1", "key2": "value2"}'
        ])
        obj = Args.parse_namespace(ns)
        self.assertEqual(obj.config, {'key1': 'value1', 'key2': 'value2'})

        # Test default value
        ns = parser.parse_args([])
        obj = Args.parse_namespace(ns)
        self.assertEqual(obj.config, {})

    def test_optional(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            name: Optional[str] = None
            age: Optional[int] = None

        parser = Args.get_parser()
        # Test specified value
        ns = parser.parse_args([
            '--name', 'Alice',
        ])
        obj = Args.parse_namespace(ns)
        self.assertEqual(obj.name, 'Alice')
        self.assertIsNone(obj.age)
        # Test specified value with age
        ns = parser.parse_args([
            '--age', '25'
        ])
        obj = Args.parse_namespace(ns)
        self.assertIsNone(obj.name)
        self.assertEqual(obj.age, 25)
        # Test default value
        ns = parser.parse_args([])
        obj = Args.parse_namespace(ns)
        self.assertIsNone(obj.name)
        self.assertIsNone(obj.age)

    def test_mixed_types(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            name: str
            age: int
            active: bool = True
            tags: List[str] = dataclasses.field(default_factory=list)
            params: Dict[str, str] = dataclasses.field(default_factory=dict)

        parser = Args.get_parser()
        # Test specified value
        ns = parser.parse_args([
            '--name', 'Alice',
            '--age', '30',
            '--no-active',
            '--tags', '["tag1", "tag2"]',
            '--params', '{"param1": "value1", "param2": "value2"}'
        ])
        obj = Args.parse_namespace(ns)
        self.assertEqual(obj.name, 'Alice')
        self.assertEqual(obj.age, 30)
        self.assertFalse(obj.active)
        self.assertEqual(obj.tags, ['tag1', 'tag2'])
        self.assertEqual(obj.params, {'param1': 'value1', 'param2': 'value2'})
        # Test default values
        ns = parser.parse_args(['--name', 'Bob', '--age', '25'])
        obj = Args.parse_namespace(ns)
        self.assertEqual(obj.name, 'Bob')
        self.assertEqual(obj.age, 25)
        self.assertTrue(obj.active)
        self.assertEqual(obj.tags, [])
        self.assertEqual(obj.params, {})

    def test_failures(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            name: str
            age: int
            tags: List[str] = dataclasses.field(default_factory=list)

        parser = Args.get_parser()
        # Missing required argument
        with self.assertRaises(ValueError):
            ns = parser.parse_args([])
            Args.parse_namespace(ns)

        # Invalid type for age
        with self.assertRaises(SystemExit):
            ns = parser.parse_args(['--name', 'Alice', '--age', 'twenty'])
            Args.parse_namespace(ns)

        # Invalid JSON for list
        with self.assertRaises(json.JSONDecodeError):
            ns = parser.parse_args([
                '--name', 'Alice', '--age', '30', '--tags', '[1, 2, 3'
            ])
            Args.parse_namespace(ns)


@auto_cli
@dataclasses.dataclass
class TestConfig:
    name: str = 'DefaultName'
    age: int = 25
    active: bool = True
    tags: List[str] = dataclasses.field(default_factory=list)
    params: Dict[str, str] = dataclasses.field(default_factory=dict)
    tags_str: List[str] = dataclasses.field(default_factory=list)
    params_str: Dict[str, str] = dataclasses.field(default_factory=dict)

class TestJsonConfig(unittest.TestCase):
    def setUp(self):
        self.cls = TestConfig

    def test_json_config(self):
        # Create a temporary JSON config file
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.json') as f:
            f.write(json.dumps({
                'name': 'Alice',
                'age': 30,
                'active': False,
                'tags': ['tag1', 'tag2'],
                'params': {'param1': 'value1', 'param2': 'value2'},
                'tags_str': '["tag3", "tag4"]',
                'params_str': '{"param3": "value3", "param4": "value4"}'
            }).encode('utf-8'))
            f.flush()
            f.seek(0)
            config_path = f.name
            # Parse command line with config file
            obj = self.cls.parse_args(['--config', config_path])
            self.assertEqual(obj.name, 'Alice')
            self.assertEqual(obj.age, 30)
            self.assertFalse(obj.active)
            self.assertEqual(obj.tags, ['tag1', 'tag2'])
            self.assertEqual(obj.params, {'param1': 'value1', 'param2': 'value2'})
            self.assertEqual(obj.tags_str, ['tag3', 'tag4'])
            self.assertEqual(obj.params_str, {'param3': 'value3', 'param4': 'value4'})

            # Test with overriding config file
            obj = self.cls.parse_args(['--config', config_path, '--active'])
            self.assertTrue(obj.active)

            obj = self.cls.parse_args(['--config', config_path, '--name', 'Bob'])
            self.assertEqual(obj.name, 'Bob')

    def test_json_config_with_defaults(self):
        # Create a temporary JSON config file with defaults
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.json') as f:
            f.write(json.dumps({}).encode('utf-8'))
            f.flush()
            f.seek(0)
            config_path = f.name

            # Parse command line with config file
            obj = self.cls.parse_args(['--config', config_path])
            self.assertEqual(obj.name, 'DefaultName')
            self.assertEqual(obj.age, 25)
            self.assertTrue(obj.active)
            self.assertEqual(obj.tags, [])
            self.assertEqual(obj.params, {})
            self.assertEqual(obj.tags_str, [])
            self.assertEqual(obj.params_str, {})


@unittest.skipIf(not parser.YAML_AVAILABLE, "YAML support not available")
class TestYamlConfig(unittest.TestCase):
    def setUp(self):
        self.cls = TestConfig

    def test_yaml_config(self):
        # Create a temporary YAML config file
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.yaml') as f:
            f.write("""
name: Alice
age: 30
active: false
tags:
    - tag1
    - tag2
params:
    param1: value1
    param2: value2
tags_str: '["tag3", "tag4"]'
params_str: '{"param3": "value3", "param4": "value4"}'
            """.strip().encode('utf-8'))
            f.flush()
            f.seek(0)
            config_path = f.name
            # Parse command line with config file
            obj = self.cls.parse_args(['--config', config_path])
            self.assertEqual(obj.name, 'Alice')
            self.assertEqual(obj.age, 30)
            self.assertFalse(obj.active)
            self.assertEqual(obj.tags, ['tag1', 'tag2'])
            self.assertEqual(obj.params, {'param1': 'value1', 'param2': 'value2'})
            self.assertEqual(obj.tags_str, ['tag3', 'tag4'])
            self.assertEqual(obj.params_str, {'param3': 'value3', 'param4': 'value4'})

            obj = self.cls.parse_args(['--config', config_path, '--active'])
            self.assertTrue(obj.active)
            obj = self.cls.parse_args(['--config', config_path, '--name', 'Bob'])
            self.assertEqual(obj.name, 'Bob')
            obj = self.cls.parse_args(['--config', config_path, '--tags', '["tag3"]'])
            self.assertEqual(obj.tags, ['tag3'])
            obj = self.cls.parse_args(['--config', config_path, '--params', '{"param3": "value3"}'])
            self.assertEqual(obj.params, {'param3': 'value3'})

    def test_yaml_config_with_defaults(self):
        # Create a temporary YAML config file with defaults
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.yaml') as f:
            f.write("{}".encode('utf-8'))
            f.flush()
            f.seek(0)
            config_path = f.name

            # Parse command line with config file
            obj = self.cls.parse_args(['--config', config_path])
            self.assertEqual(obj.name, 'DefaultName')
            self.assertEqual(obj.age, 25)
            self.assertTrue(obj.active)
            self.assertEqual(obj.tags, [])
            self.assertEqual(obj.params, {})
            self.assertEqual(obj.tags_str, [])
            self.assertEqual(obj.params_str, {})


@unittest.skipIf(not parser.TOML_AVAILABLE, "TOML support not available")
class TestTomlConfig(unittest.TestCase):
    def setUp(self):
        self.cls = TestConfig

    def test_toml_config(self):
        # Create a temporary TOML config file
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.toml') as f:
            f.write(b"""
name = "Alice"
age = 30
active = false
tags = ["tag1", "tag2"]
params = { param1 = "value1", param2 = "value2" }
tags_str = '["tag3", "tag4"]'
params_str = '{"param3": "value3", "param4": "value4"}'
            """)
            f.flush()
            f.seek(0)
            config_path = f.name
            # Parse command line with config file
            obj = self.cls.parse_args(['--config', config_path])
            self.assertEqual(obj.name, 'Alice')
            self.assertEqual(obj.age, 30)
            self.assertFalse(obj.active)
            self.assertEqual(obj.tags, ['tag1', 'tag2'])
            self.assertEqual(obj.params, {'param1': 'value1', 'param2': 'value2'})
            self.assertEqual(obj.tags_str, ['tag3', 'tag4'])
            self.assertEqual(obj.params_str, {'param3': 'value3', 'param4': 'value4'})

            obj = self.cls.parse_args(['--config', config_path, '--active'])
            self.assertTrue(obj.active)
            obj = self.cls.parse_args(['--config', config_path, '--name', 'Bob'])
            self.assertEqual(obj.name, 'Bob')
            obj = self.cls.parse_args(['--config', config_path, '--tags', '["tag3"]'])
            self.assertEqual(obj.tags, ['tag3'])
            obj = self.cls.parse_args(['--config', config_path, '--params', '{"param3": "value3"}'])
            self.assertEqual(obj.params, {'param3': 'value3'})

    def test_toml_config_with_defaults(self):
        # Create a temporary TOML config file with defaults
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.toml') as f:
            f.write(b"")
            f.flush()
            f.seek(0)
            config_path = f.name

            # Parse command line with config file
            obj = self.cls.parse_args(['--config', config_path])
            self.assertEqual(obj.name, 'DefaultName')
            self.assertEqual(obj.age, 25)
            self.assertTrue(obj.active)
            self.assertEqual(obj.tags, [])
            self.assertEqual(obj.params, {})
            self.assertEqual(obj.tags_str, [])
            self.assertEqual(obj.params_str, {})

class TestGatheredParser(unittest.TestCase):
    def test_get_all_parser(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            name: str

        @auto_cli
        @dataclasses.dataclass
        class Args1:
            age: int

        @auto_cli
        @dataclasses.dataclass
        class Args2:
            city: str

        parser = get_all_parser(Args, args1=Args1, args2=Args2)

        ns = parser.parse_args([
            '--name', 'Alice',
            '--args1-age', '30',
            '--args2-city', 'New York'
        ])
        obj = Args.parse_namespace(ns, prefix='')
        self.assertEqual(obj.name, 'Alice')
        obj1 = Args1.parse_namespace(ns, prefix='args1')
        self.assertEqual(obj1.age, 30)
        obj2 = Args2.parse_namespace(ns, prefix='args2')
        self.assertEqual(obj2.city, 'New York')

    def test_config_file(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            name: str

        @auto_cli
        @dataclasses.dataclass
        class Args1:
            age: int

        @auto_cli
        @dataclasses.dataclass
        class Args2:
            city: str

        # Create a temporary JSON config file
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.json') as f:
            f.write(json.dumps({
                'name': 'Alice',
                'args1_age': 30,
                'args2_city': 'New York'
            }).encode('utf-8'))
            config_path = f.name
            f.flush()
            f.seek(0)
            parsed = parse_all_args(
                ['--config', config_path],
                Args, args1=Args1, args2=Args2
            )
            self.assertEqual(parsed[''].name, 'Alice')
            self.assertEqual(parsed['args1'].age, 30)
            self.assertEqual(parsed['args2'].city, 'New York')

    def test_parse_all_args(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            name: str

        @auto_cli
        @dataclasses.dataclass
        class Args1:
            age: int

        @auto_cli
        @dataclasses.dataclass
        class Args2:
            city: str

        args = parse_all_args([
            '--name', 'Alice',
            '--args1-age', '30',
            '--args2-city', 'New York'
        ], Args, args1=Args1, args2=Args2)
        self.assertEqual(args[''].name, 'Alice')
        self.assertEqual(args['args1'].age, 30)
        self.assertEqual(args['args2'].city, 'New York')

    def test_parse_all_args_from_cli(self):
        @auto_cli
        @dataclasses.dataclass
        class Args:
            name: str

        @auto_cli
        @dataclasses.dataclass
        class Args1:
            age: int

        @auto_cli
        @dataclasses.dataclass
        class Args2:
            city: str


        old_argv = sys.argv
        sys.argv = [
            __file__,  # This is the script name
            '--name', 'Alice',
            '--args1-age', '30',
            '--args2-city', 'New York'
        ]

        args = parse_all_args(Args, args1=Args1, args2=Args2)
        self.assertEqual(args[''].name, 'Alice')
        self.assertEqual(args['args1'].age, 30)
        self.assertEqual(args['args2'].city, 'New York')

        sys.argv = old_argv  # Restore original argv
